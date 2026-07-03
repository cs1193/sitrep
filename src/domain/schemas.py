"""Domain entities (the heart of the model).

Entities have identity (a stable ``id``), lifecycle, and self-validating
invariants. They are plain dataclasses — persistence and transport are concerns
of the adapter/infrastructure layers.

Note: the module is named ``schemas`` per the project layout; it hosts the
:class:`Schema` aggregate and its sibling entities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.domain.value_objects import FactStatus, RetrievalResult  # noqa: F401 (re-export convenience)
from src.utils.common import count_tokens_heuristic, generate_id, utc_now, utc_now_iso


def _coerce_iso(value: Any) -> Optional[str]:
    """Coerce datetimes / strings to an ISO-8601 string (None passthrough)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"expected datetime/str/None, got {type(value).__name__}")


# =========================================================================== Schema
@dataclass
class Schema:
    """A reusable structural schema describing a class of facts.

    Schemas are promoted (marked canonical) once their usage count crosses a
    threshold, enabling the system to converge on stable structures.
    """

    name: str
    description: str = ""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    domain: str = "general"
    version: int = 1
    usage_count: int = 0
    is_promoted: bool = False
    id: str = field(default_factory=lambda: generate_id("schema"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Schema.name must be non-empty")
        if self.version < 1:
            raise ValueError("Schema.version must be >= 1")
        if self.usage_count < 0:
            raise ValueError("Schema.usage_count must be >= 0")

    def increment_usage(self, by: int = 1) -> int:
        """Bump the usage counter and return the new value."""
        if by < 0:
            raise ValueError("increment must be non-negative")
        self.usage_count += by
        return self.usage_count

    def maybe_promote(self, threshold: int) -> bool:
        """Promote this schema if usage crosses *threshold*; return new state."""
        if not self.is_promoted and self.usage_count >= threshold:
            self.is_promoted = True
        return self.is_promoted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "fields": list(self.fields),
            "domain": self.domain,
            "version": self.version,
            "usage_count": self.usage_count,
            "is_promoted": self.is_promoted,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        return cls(
            id=data.get("id") or generate_id("schema"),
            name=data["name"],
            description=data.get("description", ""),
            fields=list(data.get("fields", [])),
            domain=data.get("domain", "general"),
            version=int(data.get("version", 1)),
            usage_count=int(data.get("usage_count", 0)),
            is_promoted=bool(data.get("is_promoted", False)),
            created_at=data.get("created_at") or utc_now_iso(),
        )


# =========================================================================== Fact
@dataclass
class Fact:
    """A discrete, time-aware proposition extracted from passages.

    Bi-temporal semantics: ``valid_from`` / ``valid_to`` describe when the fact
    was true in the world; ``created_at`` describes when it was recorded.
    Facts are *invalidated* (not deleted) on update.
    """

    subject: str
    predicate: str
    object_value: str
    schema_id: Optional[str] = None
    source_passage_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    valid_from: str = field(default_factory=utc_now_iso)
    valid_to: Optional[str] = None
    status: FactStatus = FactStatus.VALID
    episode_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("fact"))
    created_at: str = field(default_factory=utc_now_iso)
    invalidated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.subject or not self.predicate or not self.object_value:
            raise ValueError("Fact.subject, predicate and object_value are required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"Fact.confidence must be in [0,1]; got {self.confidence}")
        if isinstance(self.status, str):
            self.status = FactStatus(self.status)
        self.valid_from = _coerce_iso(self.valid_from) or utc_now_iso()
        self.valid_to = _coerce_iso(self.valid_to)
        self.invalidated_at = _coerce_iso(self.invalidated_at)

    @property
    def triple(self) -> str:
        """Return the canonical ``subject | predicate | object`` string."""
        return f"{self.subject} | {self.predicate} | {self.object_value}"

    def is_valid_at(self, moment: datetime) -> bool:
        """Return True if the fact was valid at *moment* (and not invalidated)."""
        if self.status == FactStatus.INVALIDATED:
            return False
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)

        def _aware(iso: Optional[str]) -> Optional[datetime]:
            if not iso:
                return None
            dt = datetime.fromisoformat(iso)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        created = _aware(self.created_at)
        if created is not None and moment < created:
            return False
        valid_to = _aware(self.valid_to)
        if valid_to is not None and valid_to < moment:
            return False
        return True

    def invalidate(self, when: Optional[datetime] = None, reason: str = "") -> None:
        """Mark the fact invalidated as of *when* (defaults to now)."""
        moment = when or utc_now()
        self.status = FactStatus.INVALIDATED
        self.valid_to = moment.isoformat()
        self.invalidated_at = moment.isoformat()
        if reason:
            self.attributes.setdefault("invalidation_reason", reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "schema_id": self.schema_id,
            "source_passage_ids": list(self.source_passage_ids),
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "status": self.status.value,
            "episode_id": self.episode_id,
            "attributes": dict(self.attributes),
            "created_at": self.created_at,
            "invalidated_at": self.invalidated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(
            id=data.get("id") or generate_id("fact"),
            subject=data["subject"],
            predicate=data["predicate"],
            object_value=data["object_value"],
            schema_id=data.get("schema_id"),
            source_passage_ids=list(data.get("source_passage_ids", [])),
            confidence=float(data.get("confidence", 0.5)),
            valid_from=data.get("valid_from") or utc_now_iso(),
            valid_to=data.get("valid_to"),
            status=FactStatus(data.get("status", "valid")),
            episode_id=data.get("episode_id"),
            attributes=dict(data.get("attributes", {})),
            created_at=data.get("created_at") or utc_now_iso(),
            invalidated_at=data.get("invalidated_at"),
        )


# =========================================================================== TemporalFact
@dataclass
class TemporalFact(Fact):
    """A :class:`Fact` enriched with explicit bi-temporal bookkeeping.

    ``recorded_at`` captures when the system learned the fact; the inherited
    ``valid_from`` / ``valid_to`` capture validity in the world. Supports
    point-in-time queries without mutating the original record.
    """

    recorded_at: str = field(default_factory=utc_now_iso)

    def at_time(self, moment: datetime) -> Optional["TemporalFact"]:
        """Return this fact if it was both recorded and valid at *moment*."""
        recorded = datetime.fromisoformat(self.recorded_at)
        if moment < recorded:
            return None
        return self if self.is_valid_at(moment) else None

    def version_label(self) -> str:
        """Return a human-readable temporal label."""
        return f"{self.triple}  [valid {self.valid_from} → {self.valid_to or '∞'}]"


# =========================================================================== Passage
@dataclass
class Passage:
    """A unit of ingested text (a chunk) with optional embedding."""

    text: str
    document_id: str
    chunk_index: int = 0
    schema_ids: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("passage"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("Passage.text must be non-empty")
        if not self.document_id:
            raise ValueError("Passage.document_id is required")
        if self.chunk_index < 0:
            raise ValueError("Passage.chunk_index must be >= 0")
        if self.tokens <= 0:
            self.tokens = count_tokens_heuristic(self.text)

    def add_schema(self, schema_id: str) -> None:
        """Associate *schema_id* with this passage (idempotent)."""
        if schema_id and schema_id not in self.schema_ids:
            self.schema_ids.append(schema_id)

    @property
    def preview(self) -> str:
        """Return a short preview of the passage text."""
        return self.text[:160] + ("…" if len(self.text) > 160 else "")

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "schema_ids": list(self.schema_ids),
            "tokens": self.tokens,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }
        if include_embedding:
            data["embedding"] = list(self.embedding) if self.embedding else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        return cls(
            id=data.get("id") or generate_id("passage"),
            text=data["text"],
            document_id=data["document_id"],
            chunk_index=int(data.get("chunk_index", 0)),
            schema_ids=list(data.get("schema_ids", [])),
            embedding=data.get("embedding"),
            tokens=int(data.get("tokens", 0)),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at") or utc_now_iso(),
        )


# =========================================================================== Episode
@dataclass
class Episode:
    """A bounded temporal episode grouping related facts (Graphiti-style)."""

    name: str
    description: str = ""
    fact_ids: List[str] = field(default_factory=list)
    start: Optional[str] = None
    end: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("episode"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Episode.name must be non-empty")
        self.start = _coerce_iso(self.start)
        self.end = _coerce_iso(self.end)

    def add_fact(self, fact_id: str) -> None:
        """Associate *fact_id* with this episode (idempotent)."""
        if fact_id and fact_id not in self.fact_ids:
            self.fact_ids.append(fact_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "fact_ids": list(self.fact_ids),
            "start": self.start,
            "end": self.end,
            "attributes": dict(self.attributes),
            "created_at": self.created_at,
        }


# =========================================================================== Agent
@dataclass
class Agent:
    """Configuration record for a multi-agent pipeline participant."""

    name: str
    role: str
    config: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    id: str = field(default_factory=lambda: generate_id("agent"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise ValueError("Agent.name and Agent.role are required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "config": dict(self.config),
            "active": self.active,
            "created_at": self.created_at,
        }


# =========================================================================== Decision
@dataclass
class Decision:
    """An auditable decision recorded in the lineage graph."""

    agent_id: str
    decision_type: str
    action: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("decision"))
    timestamp: str = field(default_factory=utc_now_iso)
    lineage_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.agent_id or not self.decision_type or not self.action:
            raise ValueError("Decision.agent_id, decision_type and action are required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "decision_type": self.decision_type,
            "action": self.action,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "rationale": self.rationale,
            "episode_id": self.episode_id,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
            "lineage_ref": self.lineage_ref,
        }


# =========================================================================== Skill
@dataclass
class Skill:
    """A reusable, parameterized prompt-skill (evolves with feedback)."""

    name: str
    description: str = ""
    prompt: str = ""
    version: int = 1
    usage_count: int = 0
    learned_params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("skill"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Skill.name must be non-empty")
        if self.version < 1:
            raise ValueError("Skill.version must be >= 1")

    def increment_usage(self, by: int = 1) -> int:
        """Bump usage and return the new count."""
        if by < 0:
            raise ValueError("increment must be non-negative")
        self.usage_count += by
        return self.usage_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "version": self.version,
            "usage_count": self.usage_count,
            "learned_params": dict(self.learned_params),
            "created_at": self.created_at,
        }
