"""Forgetting domain: lifecycle vocabulary + decision criteria (Quantico Pattern 8).

Memory items move through a status lifecycle; forgetting is a *strategy*, not a
delete. Criteria default from Quantico (max_age 365d, inactive 180d,
min_importance 0.2, daily decay ×0.95, redundancy ≥0.85 / count 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ForgettingReason(str, Enum):
    """Why an item became a forgetting candidate."""

    REDUNDANCY = "redundancy"
    OBSOLESCENCE = "obsolescence"
    LOW_IMPORTANCE = "low_importance"
    MEMORY_PRESSURE = "memory_pressure"
    QUALITY_DEGRADATION = "quality_degradation"
    EXPIRY = "expiry"
    NONE = "none"


class ForgettingStrategy(str, Enum):
    """What to do with a candidate (none is a hard delete by default)."""

    IMMEDIATE_REMOVAL = "immediate_removal"
    GRADUAL_FADING = "gradual_fading"
    ARCHIVAL = "archival"
    SOFT_DELETE = "soft_delete"
    CONSOLIDATION = "consolidation"
    KEEP = "keep"


class MemoryItemStatus(str, Enum):
    """Lifecycle status of a memory item."""

    ACTIVE = "active"
    FADING = "fading"
    ARCHIVED = "archived"
    SOFT_DELETED = "soft_deleted"
    PERMANENTLY_DELETED = "permanently_deleted"


@dataclass
class ForgettingCriteria:
    """Thresholds that decide whether an item is a forgetting candidate."""

    max_age_days: float = 365
    inactive_period_days: float = 180
    min_importance_score: float = 0.2
    importance_decay_rate: float = 0.95      # daily multiplier
    min_access_frequency: float = 0.1
    access_decay_half_life: float = 90
    redundancy_threshold: float = 0.85       # cosine for "near-duplicate"
    min_redundancy_count: int = 3
    target_memory_size: int = 0              # 0 = no memory-pressure cap

    def reason_for(
        self,
        *,
        age_days: float,
        inactive_days: float,
        importance: float,
        access_frequency: float,
        redundancy_count: int,
        memory_size: int = 0,
    ) -> ForgettingReason:
        """Return the first matching :class:`ForgettingReason` (or NONE)."""
        if self.target_memory_size and memory_size > self.target_memory_size:
            return ForgettingReason.MEMORY_PRESSURE
        if age_days > self.max_age_days or inactive_days > self.inactive_period_days:
            return ForgettingReason.OBSOLESCENCE
        if importance < self.min_importance_score:
            return ForgettingReason.LOW_IMPORTANCE
        if access_frequency < self.min_access_frequency:
            return ForgettingReason.LOW_IMPORTANCE
        if redundancy_count >= self.min_redundancy_count:
            return ForgettingReason.REDUNDANCY
        return ForgettingReason.NONE

    @staticmethod
    def strategy_for(reason: ForgettingReason) -> ForgettingStrategy:
        """Map a reason to the default strategy (Quantico mapping)."""
        return {
            ForgettingReason.REDUNDANCY: ForgettingStrategy.CONSOLIDATION,
            ForgettingReason.OBSOLESCENCE: ForgettingStrategy.SOFT_DELETE,
            ForgettingReason.LOW_IMPORTANCE: ForgettingStrategy.GRADUAL_FADING,
            ForgettingReason.MEMORY_PRESSURE: ForgettingStrategy.ARCHIVAL,
            ForgettingReason.QUALITY_DEGRADATION: ForgettingStrategy.ARCHIVAL,
            ForgettingReason.EXPIRY: ForgettingStrategy.ARCHIVAL,
            ForgettingReason.NONE: ForgettingStrategy.KEEP,
        }[reason]
