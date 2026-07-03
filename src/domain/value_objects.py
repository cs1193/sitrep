"""Value objects and enumerations for the SITREP domain.

Value objects are immutable, self-validating, and have no identity of their own.
They encode domain invariants (e.g. confidence ∈ [0, 1], fusion weights ≥ 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- enums
class FactStatus(str, Enum):
    """Lifecycle state of a fact (bi-temporal memory)."""

    VALID = "valid"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


class ConflictType(str, Enum):
    """Categories of conflict the detection agent scans for."""

    LOGICAL = "logical"
    TEMPORAL = "temporal"
    GRANULARITY = "granularity"
    DUPLICATE = "duplicate"
    NONE = "none"


class ResolutionStrategy(str, Enum):
    """Strategies the resolution agent may apply."""

    KEEP_NEWEST = "keep_newest"
    KEEP_OLDEST = "keep_oldest"
    MERGE = "merge"
    ADJUDICATE = "adjudicate"
    KEEP_BOTH = "keep_both"


class DecisionType(str, Enum):
    """Lineage decision categories (one per major operation)."""

    INGEST = "ingest"
    QUERY = "query"
    COMPRESS = "compress"
    RETRIEVE = "retrieve"
    FEEDBACK = "feedback"
    CONFLICT = "conflict"
    VERSION = "version"


class FeedbackPolarity(str, Enum):
    """User feedback polarity."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Domain(str, Enum):
    """Zero-shot document classification domains."""

    MEDICAL = "medical"
    TECHNICAL = "technical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    GENERAL = "general"


# --------------------------------------------------------------------------- constrained scalars
@dataclass(frozen=True)
class Confidence:
    """A confidence score normalized to [0, 1]."""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TypeError("Confidence.value must be numeric")
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError(f"Confidence must lie in [0, 1]; got {self.value}")

    def __float__(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class CompressionRatio:
    """Fraction of context retained by compression, in [0, 1].

    ``reduction`` is the complementary token-saving fraction.
    """

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.value) <= 1.0:
            raise ValueError(f"CompressionRatio must lie in [0, 1]; got {self.value}")

    @property
    def reduction(self) -> float:
        """Return the fraction of tokens removed (1 - ratio)."""
        return 1.0 - self.value

    def __float__(self) -> float:
        return float(self.value)


@dataclass
class WeightTriple:
    """Three fusion weights (bm25, vector, graph); always normalized to sum 1."""

    bm25: float
    vector: float
    graph: float

    def __post_init__(self) -> None:
        for name, val in (("bm25", self.bm25), ("vector", self.vector), ("graph", self.graph)):
            if val < 0:
                raise ValueError(f"weight '{name}' must be non-negative; got {val}")
        total = self.bm25 + self.vector + self.graph
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.bm25 /= total
        self.vector /= total
        self.graph /= total

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the weights as a (bm25, vector, graph) tuple."""
        return (self.bm25, self.vector, self.graph)

    @classmethod
    def from_triple(cls, values: Tuple[float, float, float]) -> "WeightTriple":
        """Construct from a 3-tuple of non-negative weights."""
        if len(values) != 3:
            raise ValueError("expected exactly 3 weights")
        return cls(values[0], values[1], values[2])

    def to_dict(self) -> Dict[str, float]:
        return {"bm25": self.bm25, "vector": self.vector, "graph": self.graph}


@dataclass(frozen=True)
class TimeRange:
    """An inclusive [start, end] time range (timezone-aware datetimes)."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("TimeRange.end must be >= TimeRange.start")

    def contains(self, moment: datetime) -> bool:
        """Return True when *moment* lies within [start, end]."""
        return self.start <= moment <= self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Return True when *other* overlaps this range."""
        return self.start <= other.end and other.start <= self.end


# --------------------------------------------------------------------------- retrieval result
@dataclass
class RetrievalResult:
    """A single retrieved passage/fact with provenance and scores."""

    passage_id: str
    text: str
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: Optional[float] = None
    source: str = "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Return the rerank score if present, else the fused score."""
        return self.rerank_score if self.rerank_score is not None else self.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "text": self.text,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "rerank_score": self.rerank_score,
            "source": self.source,
            "metadata": self.metadata,
        }
