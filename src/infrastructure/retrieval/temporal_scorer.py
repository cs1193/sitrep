"""Temporal-decay scoring + semantic blend (Quantico Pattern 3).

Pluggable decay strategies (exponential default, half-life 180 days). Blended
into the semantic score:

    final = (1 - t_w) * semantic + t_w * temporal      t_w = 0.3 (default)
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Mapping

_logger = logging.getLogger("sitrep.retrieval.temporal")


class TemporalScorer:
    """Computes recency-decay scores and blends them with semantic scores."""

    def __init__(self, strategy: str = "exponential", half_life_days: float = 180.0) -> None:
        """Configure the decay *strategy* and *half_life_days*."""
        self.strategy = strategy
        self.half_life_days = float(half_life_days)

    def decay(self, age_days: float) -> float:
        """Return a recency score in [0, 1] for an *age_days* value."""
        a = max(0.0, float(age_days))
        if self.strategy == "linear":
            return max(0.0, 1.0 - a / max(1.0, self.half_life_days * 4.0))
        if self.strategy == "logarithmic":
            return 1.0 / (1.0 + math.log1p(a))
        if self.strategy == "power_law":
            return (1.0 + a) ** -0.5
        if self.strategy == "step":
            return 1.0 if a <= self.half_life_days else 0.0
        # exponential (default): 2^(-age / half_life)
        hl = max(1.0, self.half_life_days)
        return 2.0 ** (-a / hl)

    def score(self, age_days_by_id: Mapping[str, float]) -> Dict[str, float]:
        """Return decay scores for a mapping of id -> age_days."""
        return {k: self.decay(v) for k, v in age_days_by_id.items()}

    @staticmethod
    def blend(
        temporal: Mapping[str, float],
        semantic: Mapping[str, float],
        temporal_weight: float = 0.3,
    ) -> Dict[str, float]:
        """Blend temporal and semantic scores: ``(1-t_w)*sem + t_w*temp`` per id."""
        ids = set(temporal) | set(semantic)
        tw = max(0.0, min(1.0, float(temporal_weight)))
        return {
            i: (1.0 - tw) * float(semantic.get(i, 0.0)) + tw * float(temporal.get(i, 0.0))
            for i in ids
        }
