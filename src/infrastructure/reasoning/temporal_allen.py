"""Allen interval algebra (Phase F4): 13 relations between time intervals.

:class:`AllenRelation` + :func:`allen_relation` classify the relationship between
two intervals (start/end datetimes; ``None`` end = open-ended → +∞). Used by the
temporal-reasoning use case for "did X happen before/during/after Y" queries.
"""
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

_logger = logging.getLogger("sitrep.reasoning.allen")


class AllenRelation(str, Enum):
    """The 13 Allen interval relations."""

    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"
    CONTAINS = "contains"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


_INVERSE = {
    AllenRelation.BEFORE: AllenRelation.AFTER,
    AllenRelation.AFTER: AllenRelation.BEFORE,
    AllenRelation.MEETS: AllenRelation.MET_BY,
    AllenRelation.MET_BY: AllenRelation.MEETS,
    AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
    AllenRelation.OVERLAPPED_BY: AllenRelation.OVERLAPS,
    AllenRelation.DURING: AllenRelation.CONTAINS,
    AllenRelation.CONTAINS: AllenRelation.DURING,
    AllenRelation.STARTS: AllenRelation.STARTED_BY,
    AllenRelation.STARTED_BY: AllenRelation.STARTS,
    AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
    AllenRelation.FINISHED_BY: AllenRelation.FINISHES,
    AllenRelation.EQUALS: AllenRelation.EQUALS,
}


def _ts(dt: Optional[datetime]) -> float:
    """Return a comparable timestamp; ``None`` → +∞ (open-ended end)."""
    if dt is None:
        return float("inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=None)
    return dt.timestamp()


def inverse(relation: AllenRelation) -> AllenRelation:
    """Return the inverse of an Allen relation (A rel B  ↔  B rel⁻¹ A)."""
    return _INVERSE[relation]


def allen_relation(
    a_start: datetime,
    a_end: Optional[datetime],
    b_start: datetime,
    b_end: Optional[datetime],
) -> AllenRelation:
    """Classify the relation of interval A to interval B."""
    a_s, a_e = _ts(a_start), _ts(a_end)
    b_s, b_e = _ts(b_start), _ts(b_end)
    if a_e < b_s:
        return AllenRelation.BEFORE
    if a_s > b_e:
        return AllenRelation.AFTER
    if a_e == b_s:
        return AllenRelation.MEETS
    if a_s == b_e:
        return AllenRelation.MET_BY
    if a_s == b_s:
        if a_e == b_e:
            return AllenRelation.EQUALS
        return AllenRelation.STARTS if a_e < b_e else AllenRelation.STARTED_BY
    if a_e == b_e:
        return AllenRelation.FINISHES if a_s > b_s else AllenRelation.FINISHED_BY
    if a_s < b_s < a_e < b_e:
        return AllenRelation.OVERLAPS
    if b_s < a_s < b_e < a_e:
        return AllenRelation.OVERLAPPED_BY
    if a_s > b_s and a_e < b_e:
        return AllenRelation.DURING
    if a_s < b_s and a_e > b_e:
        return AllenRelation.CONTAINS
    # Fallback (shouldn't happen for well-formed intervals): pick by ordering.
    return AllenRelation.BEFORE if a_s < b_s else AllenRelation.AFTER
