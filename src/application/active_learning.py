"""Active learning: ask for clarification when confidence is low.

If the answer confidence falls below a threshold, the system surfaces a
clarifying question (LLM-generated when possible, else heuristic) so the user
can disambiguate before trusting the answer.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.domain.interfaces import LLMGateway
from src.domain.value_objects import RetrievalResult

_logger = logging.getLogger("sitrep.active_learning")


class ActiveLearningService:
    """Decides when to ask for clarification and produces the question."""

    def __init__(
        self,
        llm: Optional[LLMGateway] = None,
        threshold: float = 0.55,
    ) -> None:
        """Store the optional LLM and the confidence threshold."""
        self.llm = llm
        self.threshold = threshold

    def needs_clarification(self, confidence: float) -> bool:
        """Return True if *confidence* is below the active-learning threshold."""
        return float(confidence) < self.threshold

    def ask(self, query: str, results: Sequence[RetrievalResult]) -> str:
        """Return a clarifying question for *query* given retrieved *results*."""
        if self._can_llm():
            try:
                return self._llm_question(query, results)
            except Exception as exc:  # pragma: no cover
                _logger.debug("LLM clarification failed, using heuristic: %s", exc)
        return self._heuristic_question(query, results)

    # ----------------------------------------------------------------- internals
    def _can_llm(self) -> bool:
        """Return True if a non-demo LLM is available."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _llm_question(self, query: str, results: Sequence[RetrievalResult]) -> str:
        """Ask the LLM to phrase a single clarifying question."""
        context = " ".join(r.text for r in results)[:1500]
        prompt = (
            "The retrieved evidence is ambiguous for the user's question. "
            "Ask ONE concise clarifying question.\n"
            f"Question: {query}\nEvidence: {context}\nClarifying question:"
        )
        return self.llm.generate(prompt)

    @staticmethod
    def _heuristic_question(query: str, results: Sequence[RetrievalResult]) -> str:
        """Build a deterministic clarifying question from the query/results."""
        if results:
            preview = results[0].text[:80].strip()
            return f"Could you clarify which aspect of '{query}' you mean? (e.g., related to: {preview}…)"
        return f"I'm not confident about '{query}'. Could you add more detail or rephrase?"
