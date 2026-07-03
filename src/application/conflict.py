"""Conflict use case + the ConflictDetection/Resolution/Temporal agent roles.

The agent classes wrap the adapter services with repository access; the use
case orchestrates a corpus-wide conflict-resolution pass and exposes temporal
(point-in-time) queries.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from src.adapters.services.conflict import (
    Conflict,
    ConflictDetectionService,
    ConflictResolutionService,
)
from src.domain.interfaces import FactRepository, PassageRepository
from src.domain.schemas import Fact
from src.infrastructure.retrieval.temporal_retriever import TemporalRetriever

_logger = logging.getLogger("sitrep.usecase.conflict")


class ConflictDetectionAgent:
    """Scans the fact repository for conflicts."""

    def __init__(self, fact_repo: FactRepository, detector: ConflictDetectionService) -> None:
        """Wire the fact repository and detection service."""
        self.fact_repo = fact_repo
        self.detector = detector

    def scan(self) -> List[Conflict]:
        """Return all conflicts among currently-valid facts."""
        return self.detector.detect(self.fact_repo.all_valid())


class ConflictResolutionAgent:
    """Resolves conflicts, invalidating superseded facts."""

    def __init__(
        self,
        fact_repo: FactRepository,
        passage_repo: Optional[PassageRepository],
        resolver: ConflictResolutionService,
    ) -> None:
        """Wire the fact/passage repositories and resolution service."""
        self.fact_repo = fact_repo
        self.passage_repo = passage_repo
        self.resolver = resolver

    def resolve_all(self, conflicts: List[Conflict]) -> int:
        """Resolve each conflict, invalidating losers; return count resolved."""
        resolved = 0
        for conflict in conflicts:
            passages = self._evidence_for(conflict)
            resolution = self.resolver.resolve(conflict, passages)
            for fid in resolution.invalidated_fact_ids:
                self.fact_repo.invalidate(fid, reason=conflict.description)
            resolved += 1
            _logger.info(
                "resolved %s conflict '%s' (%d invalidated)",
                conflict.conflict_type.value,
                conflict.description,
                len(resolution.invalidated_fact_ids),
            )
        return resolved

    def _evidence_for(self, conflict: Conflict) -> dict:
        """Gather passage texts referenced by a conflict's facts."""
        passages: dict = {}
        if self.passage_repo is None:
            return passages
        for fact in conflict.facts:
            for pid in fact.source_passage_ids:
                passage = self.passage_repo.get(pid)
                if passage is not None:
                    passages[pid] = passage.text
        return passages


class TemporalAgent:
    """Answers point-in-time queries over the fact history."""

    def __init__(self, temporal_retriever: TemporalRetriever) -> None:
        """Wire the temporal retriever."""
        self.temporal_retriever = temporal_retriever

    def query_at(self, query: str, moment: datetime, top_k: int = 10) -> List[Fact]:
        """Return facts valid at *moment* ranked by relevance to *query*."""
        return self.temporal_retriever.retrieve_at(query, moment, top_k=top_k)


class ConflictUseCase:
    """Corpus-wide conflict detection/resolution orchestrator."""

    def __init__(
        self,
        detection_agent: ConflictDetectionAgent,
        resolution_agent: ConflictResolutionAgent,
    ) -> None:
        """Wire the detection and resolution agents."""
        self.detection_agent = detection_agent
        self.resolution_agent = resolution_agent

    def execute(self) -> dict:
        """Detect and resolve all conflicts; return a summary dict."""
        conflicts = self.detection_agent.scan()
        resolved = self.resolution_agent.resolve_all(conflicts)
        by_type: dict = {}
        for c in conflicts:
            by_type[c.conflict_type.value] = by_type.get(c.conflict_type.value, 0) + 1
        return {
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": resolved,
            "by_type": by_type,
        }
