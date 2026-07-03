"""Explanation service (Phase E1): natural-language "why it ranked/answered".

Produces a human-readable explanation of a query result from the per-channel
scores (BM25/vector/graph/PPR/density/temporal/rerank). Uses the LLM when
available, else a deterministic templated explanation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.application.dto import QueryResultDTO
from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.services.explanation")


class ExplanationService:
    """Explains a query result in natural language."""

    def __init__(self, llm: Optional[LLMGateway] = None) -> None:
        """Store an optional LLM for richer explanations."""
        self.llm = llm

    def explain(self, dto: QueryResultDTO) -> str:
        """Return an explanation string for *dto*."""
        if not dto.results:
            return "No sources were retrieved; the answer is low-confidence."
        lines = [f"Confidence {dto.confidence:.2f} · intent={dto.intent} · "
                 f"{len(dto.results)} source(s).", "Top sources:"]
        for i, r in enumerate(dto.results[:3], 1):
            sigs = self._signals(r)
            lines.append(f"  #{i} final={r.final_score:.2f} ({', '.join(sigs)})")
        base = "\n".join(lines)
        if self._can_llm():
            try:
                return self._llm_explain(dto, base)
            except Exception as exc:  # pragma: no cover
                _logger.debug("LLM explanation failed, using template: %s", exc)
        return base

    @staticmethod
    def _signals(result) -> list:
        """Collect the non-zero per-channel signals for a result."""
        sigs = []
        for name, val in (
            ("bm25", result.bm25_score),
            ("vector", result.vector_score),
            ("graph", result.graph_score),
        ):
            if val:
                sigs.append(f"{name}={val:.2f}")
        if result.rerank_score is not None:
            sigs.append(f"rerank={result.rerank_score:.2f}")
        meta = result.metadata or {}
        for key in ("ppr_score", "entity_density", "temporal_score"):
            if meta.get(key):
                sigs.append(f"{key.split('_')[0]}={meta[key]:.2f}")
        return sigs

    def _can_llm(self) -> bool:
        """Return True if a non-demo LLM is available."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _llm_explain(self, dto: QueryResultDTO, template: str) -> str:
        """Ask the LLM to phrase a concise explanation from the template + answer."""
        prompt = (
            "Explain in one or two sentences why these sources supported the answer.\n"
            f"{template}\nAnswer: {dto.answer[:200]}\nExplanation:"
        )
        return self.llm.generate(prompt)
