"""Reward model for the compression RL agent.

Compares the answer produced from *compressed* context against the answer from
*full* context. Uses a local LLM to rate similarity when available, else a
heuristic combining embedding similarity and token-reduction bonus.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from src.domain.interfaces import EmbeddingGateway, LLMGateway, RewardModel
from src.utils.common import cosine_similarity, count_tokens_heuristic

_logger = logging.getLogger("sitrep.rl.reward")


class LLMRewardModel(RewardModel):
    """Reward = answer fidelity (compressed vs full) plus a compression bonus."""

    def __init__(
        self,
        llm: Optional[LLMGateway] = None,
        embedder: Optional[EmbeddingGateway] = None,
        fidelity_weight: float = 0.7,
        reduction_weight: float = 0.3,
    ) -> None:
        """Store LLM/embedder and the reward weighting."""
        self.llm = llm
        self.embedder = embedder
        self.fidelity_weight = fidelity_weight
        self.reduction_weight = reduction_weight

    def score(
        self,
        query: str,
        compressed_answer: str,
        full_answer: str,
        context: Optional[str] = None,
    ) -> float:
        """Return a reward in [0, 1] (higher is better)."""
        fidelity = self._fidelity(compressed_answer, full_answer)
        full_tokens = count_tokens_heuristic(full_answer)
        comp_tokens = count_tokens_heuristic(compressed_answer)
        reduction = max(0.0, 1.0 - comp_tokens / max(1, full_tokens))
        reward = self.fidelity_weight * fidelity + self.reduction_weight * reduction
        return float(max(0.0, min(1.0, reward)))

    # ----------------------------------------------------------------- internals
    def _fidelity(self, compressed_answer: str, full_answer: str) -> float:
        """Rate how well the compressed answer preserves the full answer (0..1)."""
        if not compressed_answer or not full_answer:
            return 0.0
        if self._can_llm():
            try:
                return self._llm_fidelity(compressed_answer, full_answer)
            except Exception as exc:  # pragma: no cover
                _logger.debug("LLM fidelity failed, using heuristic: %s", exc)
        return self._heuristic_fidelity(compressed_answer, full_answer)

    def _can_llm(self) -> bool:
        """Return True if a non-demo LLM is available."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _llm_fidelity(self, compressed_answer: str, full_answer: str) -> float:
        """Ask the LLM to rate answer similarity on [0, 1]."""
        prompt = (
            "Rate how completely the first answer preserves the information in the "
            "second answer. Reply with ONLY a number between 0 and 1.\n"
            f"Compressed answer: {compressed_answer[:500]}\n"
            f"Full answer: {full_answer[:500]}"
        )
        raw = self.llm.generate(prompt)
        m = re.search(r"([0-1](?:\.\d+)?)", raw or "")
        if not m:
            return self._heuristic_fidelity(compressed_answer, full_answer)
        return max(0.0, min(1.0, float(m.group(1))))

    def _heuristic_fidelity(self, compressed_answer: str, full_answer: str) -> float:
        """Embedding cosine similarity (or token overlap) between the two answers."""
        if self.embedder is not None:
            return max(
                0.0,
                cosine_similarity(
                    self.embedder.embed(compressed_answer), self.embedder.embed(full_answer)
                ),
            )
        a = set(compressed_answer.lower().split())
        b = set(full_answer.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(b)
