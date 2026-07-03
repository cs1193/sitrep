"""Judge use case (Phase E2): score an answer + close the feedback loop.

Wraps :class:`LLMJudge`; ``judge_and_feedback`` converts the judge score into a
fusion-weight update (positive nudge if the answer scored well, negative if not)
so the retriever self-improves from its own judged quality — the LLM-judge
feedback loop.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.adapters.services.judge import LLMJudge

_logger = logging.getLogger("sitrep.usecase.judge")


class JudgeUseCase:
    """Scores answers and feeds the judge signal back into fusion weights."""

    def __init__(self, judge: LLMJudge, retriever: Any = None) -> None:
        """Wire the judge + the retriever (for fusion-weight updates)."""
        self._judge = judge
        self.retriever = retriever

    def judge(self, dto) -> Dict[str, Any]:
        """Score a query result's answer against its retrieved context."""
        context = "\n".join(r.text for r in getattr(dto, "results", []))
        return self._judge.score(dto.query, dto.answer, context)

    def judge_and_feedback(self, dto) -> Dict[str, Any]:
        """Judge *dto* and apply the score as fusion feedback (the loop)."""
        result = self.judge(dto)
        feedback_applied = False
        if self.retriever is not None and getattr(dto, "results", None):
            sign = 1.0 if result["score"] >= 0.5 else -1.0
            row = {
                r.passage_id: (float(r.bm25_score), float(r.vector_score), float(r.graph_score), sign)
                for r in dto.results
            }
            if row:
                try:
                    self.retriever.fusion.update_online([row])
                    self.retriever.update_weights(self.retriever.weights)
                    feedback_applied = True
                except Exception as exc:  # pragma: no cover
                    _logger.warning("judge fusion feedback failed: %s", exc)
        result["feedback_applied"] = feedback_applied
        return result
