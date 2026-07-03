"""Consolidation use case: merge near-duplicate passages (Phase B / Pattern 8).

Finds pairs of passages with embedding cosine ≥ θ (default 0.85), keeps the
higher-importance one as canonical, and SOFT_DELETEs the other (never hard
delete) — recording a SUPERSEDES-style decision in lineage. Bounded by *limit*
to keep the O(limit × corpus) scan tractable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.adapters.services.importance import ImportanceScorer
from src.domain.interfaces import PassageRepository
from src.domain.schemas import Decision
from src.infrastructure.lineage import LineageTracker
from src.utils.constants import DEC_CONFLICT

_logger = logging.getLogger("sitrep.usecase.consolidation")


def _age_days(passage) -> float:
    """Return passage age in days from its ``created_at`` ISO string."""
    try:
        created = datetime.fromisoformat(passage.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400.0)
    except Exception:
        return 0.0


class ConsolidationUseCase:
    """Merge near-duplicate passages into a single canonical passage."""

    def __init__(
        self,
        passage_repo: PassageRepository,
        importance_scorer: ImportanceScorer,
        lineage_tracker: LineageTracker,
        theta: float = 0.85,
        max_pairs: int = 200,
    ) -> None:
        """Wire the passage repo, importance scorer, lineage, and thresholds."""
        self.passage_repo = passage_repo
        self.importance_scorer = importance_scorer
        self.lineage_tracker = lineage_tracker
        self.theta = float(theta)
        self.max_pairs = int(max_pairs)

    def execute(self, limit: Optional[int] = None) -> dict:
        """Scan up to *limit* passages, merging near-duplicates; return a report."""
        cap = int(limit or self.max_pairs)
        passages = list(self.passage_repo.iter_with_embeddings())
        if not passages:
            return {"examined": 0, "merged": 0, "theta": self.theta}
        max_ac = max((int(p.metadata.get("access_count", 0)) for p in passages), default=1) or 1

        examined = merged = 0
        for passage in passages[:cap]:
            examined += 1
            if str(passage.metadata.get("memory_status", "active")) != "active":
                continue
            if passage.embedding is None:
                continue
            try:
                candidates = self.passage_repo.find_near_duplicates(
                    passage.embedding, self.theta, limit=6
                )
            except Exception as exc:  # pragma: no cover
                _logger.warning("find_near_duplicates failed: %s", exc)
                continue
            imp_a = self.importance_scorer.score_passage(
                passage, age_days=_age_days(passage), max_access_count=max_ac
            )
            for other, sim in candidates:
                if other.id == passage.id:
                    continue
                if str(other.metadata.get("memory_status", "active")) != "active":
                    continue
                imp_b = self.importance_scorer.score_passage(
                    other, age_days=_age_days(other), max_access_count=max_ac
                )
                canonical, loser = (passage, other) if imp_a >= imp_b else (other, passage)
                loser.metadata["memory_status"] = "soft_deleted"
                loser.metadata["consolidated_into"] = canonical.id
                self.passage_repo.save(loser)
                self.lineage_tracker.record(
                    Decision(
                        agent_id="consolidation",
                        decision_type=DEC_CONFLICT,
                        action="consolidate",
                        inputs={"pair": [passage.id, other.id], "sim": round(float(sim), 3)},
                        outputs={"canonical": canonical.id, "loser": loser.id},
                        rationale="near-duplicate merge (loser SOFT_DELETED)",
                    )
                )
                merged += 1
                break  # one consolidation per passage per run
        _logger.info("consolidation: examined=%d merged=%d", examined, merged)
        return {"examined": examined, "merged": merged, "theta": self.theta}
