"""Forgetting use case: classify items and apply lifecycle strategies (Phase B).

Daily importance decay + candidate classification via :class:`ForgettingCriteria`.
Defaults are non-destructive (``dry_run=True``, SOFT_DELETE/ARCHIVAL/FADING); a
hard delete requires an explicit strategy override (PERMANENTLY_DELETED).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.adapters.services.importance import ImportanceScorer
from src.domain.forgetting import ForgettingCriteria, ForgettingStrategy, MemoryItemStatus
from src.domain.interfaces import PassageRepository
from src.domain.schemas import Decision
from src.infrastructure.lineage import LineageTracker

_logger = logging.getLogger("sitrep.usecase.forgetting")

_INACTIVE_STATUSES = {
    MemoryItemStatus.SOFT_DELETED.value,
    MemoryItemStatus.ARCHIVED.value,
    MemoryItemStatus.PERMANENTLY_DELETED.value,
}


def _age_days(passage) -> float:
    """Return age in days from ``created_at``."""
    try:
        created = datetime.fromisoformat(passage.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def _inactive_days(passage) -> float:
    """Return inactive days (from ``last_accessed_at`` if present, else age)."""
    last = passage.metadata.get("last_accessed_at")
    if not last:
        return _age_days(passage)
    try:
        la = datetime.fromisoformat(last)
        if la.tzinfo is None:
            la = la.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - la).total_seconds() / 86400.0)
    except Exception:
        return _age_days(passage)


class ForgettingUseCase:
    """Classify passages by the forgetting criteria and apply lifecycle actions."""

    def __init__(
        self,
        passage_repo: PassageRepository,
        importance_scorer: ImportanceScorer,
        lineage_tracker: LineageTracker,
        criteria: ForgettingCriteria,
    ) -> None:
        """Wire the passage repo, importance scorer, lineage, and criteria."""
        self.passage_repo = passage_repo
        self.importance_scorer = importance_scorer
        self.lineage_tracker = lineage_tracker
        self.criteria = criteria

    def execute(self, dry_run: bool = True, memory_size: Optional[int] = None) -> dict:
        """Classify every active passage; apply strategies unless *dry_run*."""
        passages = list(self.passage_repo.iter_all())
        active = [p for p in passages if str(p.metadata.get("memory_status", "active")) not in _INACTIVE_STATUSES]
        max_ac = max((int(p.metadata.get("access_count", 0)) for p in active), default=1) or 1
        size = memory_size if memory_size is not None else len(active)
        report: dict = {
            "examined": 0,
            "dry_run": bool(dry_run),
            "active": len(active),
            "memory_size": size,
            "by_strategy": {},
            "actions": [],
        }

        for passage in active:
            report["examined"] += 1
            age = _age_days(passage)
            inactive = _inactive_days(passage)
            access_frequency = float(passage.metadata.get("access_count", 0))
            importance = float(
                passage.metadata.get("importance")
                or self.importance_scorer.score_passage(
                    passage, age_days=age, max_access_count=max_ac
                )
            )
            redundancy = int(passage.metadata.get("redundancy_count", 0))
            reason = self.criteria.reason_for(
                age_days=age,
                inactive_days=inactive,
                importance=importance,
                access_frequency=access_frequency,
                redundancy_count=redundancy,
                memory_size=size,
            )
            strategy = ForgettingCriteria.strategy_for(reason)
            report["by_strategy"][strategy.value] = report["by_strategy"].get(strategy.value, 0) + 1
            if strategy == ForgettingStrategy.KEEP:
                continue
            action = {
                "id": passage.id,
                "reason": reason.value,
                "strategy": strategy.value,
                "importance": round(importance, 3),
                "age_days": round(age, 1),
            }
            report["actions"].append(action)
            if not dry_run:
                self._apply(passage, strategy)
                self.passage_repo.save(passage)
            self.lineage_tracker.record(
                Decision(
                    agent_id="forgetting",
                    decision_type="forget",
                    action="forget_item",
                    inputs={"id": passage.id, "age_days": round(age, 1)},
                    outputs=action,
                    rationale=f"{reason.value} -> {strategy.value}",
                )
            )
        _logger.info("forgetting: examined=%d actions=%d dry_run=%s", report["examined"], len(report["actions"]), dry_run)
        return report

    def decay_all(self, dry_run: bool = False) -> dict:
        """Apply the daily importance decay (×``importance_decay_rate``) to active/fading items."""
        passages = list(self.passage_repo.iter_all())
        decayed = 0
        for passage in passages:
            status = str(passage.metadata.get("memory_status", "active"))
            if status in _INACTIVE_STATUSES:
                continue
            current = float(passage.metadata.get("importance", 0.5))
            passage.metadata["importance"] = round(current * self.criteria.importance_decay_rate, 4)
            if not dry_run:
                self.passage_repo.save(passage)
            decayed += 1
        return {"decayed": decayed, "rate": self.criteria.importance_decay_rate, "dry_run": bool(dry_run)}

    @staticmethod
    def _apply(passage, strategy: ForgettingStrategy) -> None:
        """Mutate the passage's metadata according to *strategy* (non-destructive)."""
        if strategy == ForgettingStrategy.SOFT_DELETE:
            passage.metadata["memory_status"] = MemoryItemStatus.SOFT_DELETED.value
        elif strategy == ForgettingStrategy.ARCHIVAL:
            passage.metadata["memory_status"] = MemoryItemStatus.ARCHIVED.value
        elif strategy == ForgettingStrategy.GRADUAL_FADING:
            current = float(passage.metadata.get("importance", 0.5))
            passage.metadata["importance"] = round(current * 0.95, 4)
            passage.metadata["memory_status"] = MemoryItemStatus.FADING.value
        elif strategy == ForgettingStrategy.IMMEDIATE_REMOVAL:
            passage.metadata["memory_status"] = MemoryItemStatus.PERMANENTLY_DELETED.value
        elif strategy == ForgettingStrategy.CONSOLIDATION:
            passage.metadata["consolidation_candidate"] = True
