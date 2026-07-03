"""Data Transfer Objects (DTOs) crossing the application boundary.

DTOs are plain data carriers with no behavior; they decouple use-case results
from domain entities and from any specific serialization format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.domain.value_objects import RetrievalResult


@dataclass
class IngestResultDTO:
    """Summary of a document ingestion operation."""

    document_id: str
    passages: int = 0
    facts: int = 0
    schemas: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    domain: str = "general"
    method: str = "regex"
    episode_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "passages": self.passages,
            "facts": self.facts,
            "schemas": self.schemas,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "domain": self.domain,
            "method": self.method,
            "episode_id": self.episode_id,
        }


@dataclass
class QueryResultDTO:
    """Full result of a knowledge query (answer + provenance + telemetry)."""

    query_id: str
    query: str
    answer: str
    results: Sequence[RetrievalResult] = field(default_factory=list)
    confidence: float = 0.0
    quality: float = 0.0
    compression_ratio: float = 1.0
    full_tokens: int = 0
    compressed_tokens: int = 0
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    backend: str = "demo"
    intent: str = "simple"
    extras: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None
    cached: bool = False
    # Headroom telemetry (content-aware compression + cache alignment + CCR)
    content_type: str = "text"
    compressor: str = "extractive"
    ccr_key: Optional[str] = None
    cache_eligible: Optional[bool] = None

    @property
    def token_reduction(self) -> float:
        """Fraction of tokens removed by compression (0..1)."""
        if self.full_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - self.compressed_tokens / self.full_tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "quality": round(self.quality, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "token_reduction": round(self.token_reduction, 4),
            "full_tokens": self.full_tokens,
            "compressed_tokens": self.compressed_tokens,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "backend": self.backend,
            "intent": self.intent,
            "extras": self.extras,
            "explanation": self.explanation,
            "cached": self.cached,
            "content_type": self.content_type,
            "compressor": self.compressor,
            "ccr_key": self.ccr_key,
            "cache_eligible": self.cache_eligible,
            "sources": [r.to_dict() for r in self.results],
        }


@dataclass
class FeedbackDTO:
    """Acknowledgement of recorded user feedback."""

    feedback_id: str
    query_id: str
    polarity: str
    rating: float
    weights_updated: bool = False
    new_weights: Optional[Sequence[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "query_id": self.query_id,
            "polarity": self.polarity,
            "rating": self.rating,
            "weights_updated": self.weights_updated,
            "new_weights": list(self.new_weights) if self.new_weights else None,
        }


@dataclass
class TrainResultDTO:
    """Outcome of an RL training run."""

    timesteps: int
    backend: str
    policy_path: Optional[str]
    mean_reward: float = 0.0
    episodes_evaluated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timesteps": self.timesteps,
            "backend": self.backend,
            "policy_path": self.policy_path,
            "mean_reward": round(self.mean_reward, 4),
            "episodes_evaluated": self.episodes_evaluated,
        }


@dataclass
class VersionDTO:
    """Metadata for a stored snapshot."""

    name: str
    path: str
    size_mb: float
    created_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionDTO":
        return cls(
            name=data["name"],
            path=data["path"],
            size_mb=float(data.get("size_mb", 0.0)),
            created_at=data.get("created_at", ""),
        )


@dataclass
class LineageTraceDTO:
    """A decision trace with graph neighbors."""

    decision: Optional[Dict[str, Any]] = None
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "neighbors": self.neighbors,
            "fetched_at": self.fetched_at,
        }


@dataclass
class StatsDTO:
    """Aggregate system statistics for the Stats UI."""

    schemas: int = 0
    facts: int = 0
    passages: int = 0
    episodes: int = 0
    decisions: int = 0
    feedback: int = 0
    kv_caches: int = 0
    fusion_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3)
    token_reduction_ratio: float = 0.0
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemas": self.schemas,
            "facts": self.facts,
            "passages": self.passages,
            "episodes": self.episodes,
            "decisions": self.decisions,
            "feedback": self.feedback,
            "kv_caches": self.kv_caches,
            "fusion_weights": list(self.fusion_weights),
            "token_reduction_ratio": round(self.token_reduction_ratio, 4),
            "tokens_saved": self.tokens_saved,
        }
