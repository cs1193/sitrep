"""Lineage tracker.

Records :class:`Decision` objects to the SQLite decision table and (when a
graph store is wired) to the Kuzu lineage graph, then supports trace queries.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.domain.interfaces import DecisionRepository, GraphStore
from src.domain.schemas import Decision
from src.utils.common import utc_now_iso

_logger = logging.getLogger("sitrep.lineage")


class LineageTracker:
    """Append-only decision ledger with graph-backed trace queries."""

    def __init__(
        self,
        decision_repo: DecisionRepository,
        graph_store: Optional[GraphStore] = None,
    ) -> None:
        """Wire the SQLite decision repository and an optional lineage graph."""
        self.decision_repo = decision_repo
        self.graph_store = graph_store

    def record(self, decision: Decision) -> str:
        """Persist *decision* to SQLite and (optionally) the lineage graph."""
        decision_id = self.decision_repo.save(decision)
        if self.graph_store is not None:
            try:
                props = {
                    "id": decision.id,
                    "agent_id": decision.agent_id,
                    "decision_type": decision.decision_type,
                    "action": decision.action,
                    "timestamp": decision.timestamp,
                }
                self.graph_store.add_entity("Decision", props)
                for cause in decision.metadata.get("caused_by", []) or []:
                    self.graph_store.add_relation(
                        "Decision", str(cause), "TriggeredBy", "Decision", decision.id
                    )
            except Exception as exc:  # pragma: no cover
                _logger.debug("lineage graph write skipped: %s", exc)
        _logger.info("lineage: recorded %s (%s)", decision_id, decision.decision_type)
        return decision_id

    def get_trace(self, decision_id: str) -> Dict[str, Any]:
        """Return a decision plus its graph neighbors (if available)."""
        decision = self.decision_repo.get(decision_id)
        neighbors: List[Dict[str, Any]] = []
        if self.graph_store is not None:
            try:
                neighbors = self.graph_store.neighbors("Decision", decision_id, limit=20)
            except Exception as exc:  # pragma: no cover
                _logger.debug("lineage neighbor query failed: %s", exc)
        return {
            "decision": decision.to_dict() if decision else None,
            "graph_neighbors": neighbors,
            "fetched_at": utc_now_iso(),
        }

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent decisions as dicts."""
        return [d.to_dict() for d in self.decision_repo.list_recent(limit)]

    def by_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """Return all decisions in an episode."""
        return [d.to_dict() for d in self.decision_repo.by_episode(episode_id)]
