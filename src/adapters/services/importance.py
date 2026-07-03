"""Importance scorer (Phase B / Quantico Pattern 8).

Composite ``importance ∈ [0,1]`` from recency, access frequency, redundancy, and
source support. The score is stashed on the passage's ``metadata`` so it survives
persistence without a schema change.
"""
from __future__ import annotations

import logging
from typing import Any

from src.domain.schemas import Passage

_logger = logging.getLogger("sitrep.services.importance")


class ImportanceScorer:
    """Weights several weak signals into a single importance score."""

    def __init__(
        self,
        recency_w: float = 0.3,
        frequency_w: float = 0.3,
        redundancy_w: float = 0.2,
        source_w: float = 0.2,
        redundancy_k: int = 5,
        max_age_days: float = 365.0,
    ) -> None:
        """Configure per-signal weights (normalized to sum 1) and caps."""
        total = recency_w + frequency_w + redundancy_w + source_w
        self.recency_w = recency_w / total
        self.frequency_w = frequency_w / total
        self.redundancy_w = redundancy_w / total
        self.source_w = source_w / total
        self.redundancy_k = max(1, int(redundancy_k))
        self.max_age_days = max(1.0, float(max_age_days))

    def score(
        self,
        *,
        access_count: int,
        max_access_count: int,
        age_days: float,
        redundancy_count: int,
        source_count: int,
    ) -> float:
        """Return an importance score in [0, 1] from the raw signals."""
        recency = max(0.0, 1.0 - float(age_days) / self.max_age_days)
        frequency = min(1.0, float(access_count) / max(1.0, float(max_access_count)))
        redundancy = 1.0 - min(1.0, float(redundancy_count) / self.redundancy_k)
        source = min(1.0, float(source_count) / 5.0)
        return (
            self.recency_w * recency
            + self.frequency_w * frequency
            + self.redundancy_w * redundancy
            + self.source_w * source
        )

    def score_passage(
        self,
        passage: Passage,
        *,
        age_days: float,
        max_access_count: int = 1,
        redundancy_count: int = 0,
    ) -> float:
        """Score *passage*, stash the result in ``metadata['importance']``."""
        access_count = int(passage.metadata.get("access_count", 0))
        source_count = len(passage.schema_ids) or int(passage.metadata.get("source_count", 1))
        importance = self.score(
            access_count=access_count,
            max_access_count=max(1, max_access_count),
            age_days=age_days,
            redundancy_count=int(passage.metadata.get("redundancy_count", redundancy_count)),
            source_count=max(1, source_count),
        )
        importance = round(float(importance), 4)
        passage.metadata["importance"] = importance
        return importance


def get_metadata(passage: Passage, key: str, default: Any = None) -> Any:
    """Read a memory-hygiene field from a passage's metadata (helper)."""
    return passage.metadata.get(key, default)
