"""Lineage analysis use case: inspect decision traces."""
from __future__ import annotations

import logging
from typing import List

from src.application.dto import LineageTraceDTO
from src.infrastructure.lineage import LineageTracker

_logger = logging.getLogger("sitrep.usecase.lineage")


class LineageAnalysisUseCase:
    """Read-only views over the lineage graph and decision ledger."""

    def __init__(self, lineage_tracker: LineageTracker) -> None:
        """Wire the lineage tracker."""
        self.lineage_tracker = lineage_tracker

    def trace(self, decision_id: str) -> LineageTraceDTO:
        """Return a full decision trace (decision + graph neighbors)."""
        data = self.lineage_tracker.get_trace(decision_id)
        return LineageTraceDTO(
            decision=data.get("decision"),
            neighbors=list(data.get("graph_neighbors", [])),
            fetched_at=data.get("fetched_at", ""),
        )

    def recent(self, limit: int = 50) -> List[dict]:
        """Return the most recent decisions."""
        return self.lineage_tracker.recent(limit)

    def by_episode(self, episode_id: str) -> List[dict]:
        """Return all decisions within an episode."""
        return self.lineage_tracker.by_episode(episode_id)
