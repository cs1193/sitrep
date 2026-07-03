"""Temporal reasoning use case (Phase F4): Allen relations between facts.

Given two fact ids, computes the Allen interval relation between their validity
windows (``valid_from`` → ``valid_to``) — answering "did X hold before/during/
after Y" over bi-temporal memory.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.domain.interfaces import FactRepository
from src.infrastructure.reasoning.temporal_allen import AllenRelation, allen_relation, inverse

_logger = logging.getLogger("sitrep.usecase.temporal_reasoning")


def _parse(iso: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string (or None → open-ended)."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


class TemporalReasoningUseCase:
    """Answers Allen-relation questions between two facts."""

    def __init__(self, fact_repo: FactRepository) -> None:
        """Wire the fact repository."""
        self.fact_repo = fact_repo

    def relate(self, fact_a_id: str, fact_b_id: str) -> Dict[str, Any]:
        """Return the Allen relation of fact A to fact B (plus the inverse)."""
        a = self.fact_repo.get(fact_a_id)
        b = self.fact_repo.get(fact_b_id)
        if a is None or b is None:
            return {"error": "one or both facts not found", "a": fact_a_id, "b": fact_b_id}
        relation = allen_relation(_parse(a.valid_from), _parse(a.valid_to),
                                  _parse(b.valid_from), _parse(b.valid_to))
        return {
            "a": {"id": a.id, "triple": a.triple, "valid_from": a.valid_from, "valid_to": a.valid_to},
            "b": {"id": b.id, "triple": b.triple, "valid_from": b.valid_from, "valid_to": b.valid_to},
            "relation": relation.value,
            "inverse": inverse(relation).value,
        }
