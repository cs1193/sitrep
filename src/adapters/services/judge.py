"""LLM-as-judge (Phase E2): score an answer for the feedback loop.

When a real LLM is available, asks it to rate the answer against the context.
Otherwise uses a deterministic heuristic (query-term coverage + answer length).
The score drives automatic fusion-weight feedback in :class:`JudgeUseCase`.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from src.domain.interfaces import EmbeddingGateway, LLMGateway
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.services.judge")


class LLMJudge:
    """Scores answer quality in [0, 1] with a rationale."""

    def __init__(self, llm: Optional[LLMGateway] = None, embedder: Optional[EmbeddingGateway] = None) -> None:
        """Store optional LLM + embedder."""
        self.llm = llm
        self.embedder = embedder

    def score(self, query: str, answer: str, context: str = "") -> Dict[str, Any]:
        """Return ``{"score": float, "rationale": str}``."""
        if not answer:
            return {"score": 0.0, "rationale": "no answer produced"}
        if self._can_llm():
            try:
                return self._llm_score(query, answer, context)
            except Exception as exc:  # pragma: no cover
                _logger.debug("LLM judge failed, using heuristic: %s", exc)
        return self._heuristic(query, answer, context)

    def _can_llm(self) -> bool:
        """Return True if a non-demo LLM is available."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _llm_score(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Ask the LLM to rate the answer 0..1 against the context."""
        prompt = (
            "Rate how well the answer is supported by the context. "
            "Reply with ONLY a number between 0 and 1 and a short reason.\n"
            f"Question: {query}\nContext: {context[:1000]}\nAnswer: {answer[:500]}\nRating:"
        )
        raw = self.llm.generate(prompt)
        m = re.search(r"([0-1](?:\.\d+)?)", raw or "")
        score = float(m.group(1)) if m else 0.5
        return {"score": round(max(0.0, min(1.0, score)), 3), "rationale": (raw or "")[:160]}

    def _heuristic(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Deterministic score from query-term coverage + answer length + context fit."""
        qterms = {t for t in (query or "").lower().split() if len(t) > 2}
        aterms = set((answer or "").lower().split())
        coverage = (len(qterms & aterms) / len(qterms)) if qterms else 0.5
        length = min(1.0, len((answer or "").split()) / 20.0)
        context_fit = 0.0
        if context and self.embedder is not None:
            try:
                context_fit = max(0.0, cosine_similarity(self.embedder.embed(answer), self.embedder(context[:500])))
            except Exception:  # pragma: no cover
                context_fit = 0.0
        score = 0.45 * coverage + 0.25 * length + 0.30 * context_fit
        return {
            "score": round(score, 3),
            "rationale": f"coverage={coverage:.2f} length={length:.2f} context_fit={context_fit:.2f}",
        }
