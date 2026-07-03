"""Conflict detection and resolution services.

Detects logical, temporal, granularity, and duplicate conflicts among facts,
then resolves them using evidence (confidence, recency, passages) with optional
LLM adjudication.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.domain.schemas import Fact
from src.domain.value_objects import ConflictType, ResolutionStrategy
from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.services.conflict")


@dataclass
class Conflict:
    """A detected conflict among two or more facts sharing a key."""

    conflict_type: ConflictType
    facts: List[Fact]
    key: str = ""
    description: str = ""


@dataclass
class ResolutionResult:
    """The outcome of resolving a :class:`Conflict`."""

    strategy: ResolutionStrategy
    kept_fact_ids: List[str] = field(default_factory=list)
    invalidated_fact_ids: List[str] = field(default_factory=list)
    merged_fact: Optional[Fact] = None
    rationale: str = ""


class ConflictDetectionService:
    """Scans a fact set for logical/temporal/granularity/duplicate conflicts."""

    def detect(self, facts: List[Fact]) -> List[Conflict]:
        """Group facts by (subject, predicate) and classify intra-group conflicts."""
        groups: Dict[str, List[Fact]] = defaultdict(list)
        for f in facts:
            if f.status.value != "valid":
                continue
            groups[f"{f.subject.lower()}||{f.predicate.lower()}"].append(f)
        conflicts: List[Conflict] = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            objects = {f.object_value.lower() for f in group}
            subject_pred = key.replace("||", "/")
            if len(objects) == 1:
                conflicts.append(
                    Conflict(
                        ConflictType.DUPLICATE, list(group), key,
                        f"Duplicate facts for {subject_pred}",
                    )
                )
                continue
            # Distinct objects → at least a logical conflict; refine type.
            if self._is_granularity(group):
                ctype = ConflictType.GRANULARITY
                desc = f"Granularity mismatch for {subject_pred}"
            elif self._has_temporal_overlap(group):
                ctype = ConflictType.TEMPORAL
                desc = f"Temporal conflict for {subject_pred} (overlapping validity)"
            else:
                ctype = ConflictType.LOGICAL
                desc = f"Logical conflict for {subject_pred}"
            conflicts.append(Conflict(ctype, list(group), key, desc))
        return conflicts

    @staticmethod
    def _is_granularity(group: List[Fact]) -> bool:
        """Return True if one object string contains another (more/less specific)."""
        objs = [f.object_value.lower() for f in group]
        for a in objs:
            for b in objs:
                if a is not b and a in b and a != b:
                    return True
        return False

    @staticmethod
    def _has_temporal_overlap(group: List[Fact]) -> bool:
        """Return True if any two facts have overlapping valid_from/valid_to windows."""

        def _range(f: Fact):
            start = (
                datetime.fromisoformat(f.valid_from).astimezone(timezone.utc)
                if f.valid_from
                else datetime.min.replace(tzinfo=timezone.utc)
            )
            end = (
                datetime.fromisoformat(f.valid_to).astimezone(timezone.utc)
                if f.valid_to
                else datetime.max.replace(tzinfo=timezone.utc)
            )
            return start, end

        ranges = [_range(f) for f in group]
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                s1, e1 = ranges[i]
                s2, e2 = ranges[j]
                if s1 <= e2 and s2 <= e1:
                    return True
        return False


class ConflictResolutionService:
    """Resolves conflicts using evidence and optional LLM adjudication."""

    def __init__(self, llm_gateway: Optional[LLMGateway] = None) -> None:
        """Store the optional LLM gateway for close-call adjudication."""
        self.llm = llm_gateway

    def resolve(
        self,
        conflict: Conflict,
        passage_texts: Optional[Dict[str, str]] = None,
    ) -> ResolutionResult:
        """Pick a strategy and return kept/invalidated fact ids."""
        passage_texts = passage_texts or {}
        facts = sorted(conflict.facts, key=lambda f: f.confidence, reverse=True)

        if conflict.conflict_type == ConflictType.DUPLICATE:
            kept = facts[0]
            return ResolutionResult(
                ResolutionStrategy.KEEP_NEWEST,
                kept_fact_ids=[kept.id],
                invalidated_fact_ids=[f.id for f in facts[1:]],
                rationale="Duplicate facts; kept highest-confidence instance.",
            )

        if conflict.conflict_type == ConflictType.GRANULARITY:
            # Keep the most specific (longest object) fact.
            most_specific = max(facts, key=lambda f: len(f.object_value))
            return ResolutionResult(
                ResolutionStrategy.KEEP_BOTH,
                kept_fact_ids=[f.id for f in facts],
                invalidated_fact_ids=[],
                rationale=f"Granularity differs; retained all (most specific: {most_specific.object_value}).",
            )

        # Logical / temporal: prefer newest, but adjudicate close calls.
        by_time = sorted(conflict.facts, key=lambda f: f.valid_from or "", reverse=True)
        newest = by_time[0]
        if self._close_call(facts) and self._can_adjudicate():
            picked = self._adjudicate(conflict, passage_texts) or newest
        else:
            picked = newest
        return ResolutionResult(
            ResolutionStrategy.KEEP_NEWEST,
            kept_fact_ids=[picked.id],
            invalidated_fact_ids=[f.id for f in conflict.facts if f.id != picked.id],
            rationale=f"Kept newest/most-confident fact ({picked.object_value}); invalidated superseded values.",
        )

    @staticmethod
    def _close_call(facts: List[Fact]) -> bool:
        """Return True if top-2 confidences differ by less than 0.1."""
        if len(facts) < 2:
            return False
        return abs(facts[0].confidence - facts[1].confidence) < 0.1

    def _can_adjudicate(self) -> bool:
        """Return True if a non-demo LLM is available for adjudication."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _adjudicate(self, conflict: Conflict, passages: Dict[str, str]) -> Optional[Fact]:
        """Ask the LLM which fact is best supported by the evidence passages."""
        evidence = "\n".join(passages.values())[:2000] if passages else "(no passages)"
        options = "\n".join(
            f"{i+1}. {f.subject} {f.predicate} {f.object_value} (conf={f.confidence:.2f})"
            for i, f in enumerate(conflict.facts)
        )
        prompt = (
            "Given the evidence, which statement is most accurate? Reply with ONLY the number.\n"
            f"Options:\n{options}\n\nEvidence:\n{evidence}"
        )
        try:
            raw = self.llm.generate(prompt)
        except Exception as exc:  # pragma: no cover
            _logger.warning("adjudication failed: %s", exc)
            return None
        import re

        m = re.search(r"(\d+)", raw or "")
        if not m:
            return None
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(conflict.facts):
            return conflict.facts[idx]
        return None
