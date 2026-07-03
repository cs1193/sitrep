"""Domain events published through the in-process :class:`EventBus`.

Events are published by use cases and consumed by cross-cutting concerns
(metrics, lineage, logging) without tight coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from src.infrastructure.event_bus import get_event_bus


def _now() -> str:
    """Return the current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DomainEvent:
    """Base event carrying a topic, payload, and timestamp."""

    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def publish(self) -> None:
        """Publish this event on the global event bus."""
        get_event_bus().publish(self.topic, {"payload": self.payload, "timestamp": self.timestamp})


# --------------------------------------------------------------------------- concrete events
def document_ingested(document_id: str, facts: int, passages: int, domain: str) -> DomainEvent:
    """Construct and return a ``document.ingested`` event."""
    return DomainEvent(
        "document.ingested",
        {"document_id": document_id, "facts": facts, "passages": passages, "domain": domain},
    )


def query_answered(
    query_id: str, query: str, confidence: float, token_reduction: float
) -> DomainEvent:
    """Construct and return a ``query.answered`` event."""
    return DomainEvent(
        "query.answered",
        {
            "query_id": query_id,
            "query": query,
            "confidence": confidence,
            "token_reduction": token_reduction,
        },
    )


def feedback_received(query_id: str, polarity: str, rating: float) -> DomainEvent:
    """Construct and return a ``feedback.received`` event."""
    return DomainEvent(
        "feedback.received",
        {"query_id": query_id, "polarity": polarity, "rating": rating},
    )


def agent_trained(backend: str, timesteps: int, mean_reward: float) -> DomainEvent:
    """Construct and return an ``agent.trained`` event."""
    return DomainEvent(
        "agent.trained",
        {"backend": backend, "timesteps": timesteps, "mean_reward": mean_reward},
    )


def conflict_resolved(conflict_type: str, kept: int, invalidated: int) -> DomainEvent:
    """Construct and return a ``conflict.resolved`` event."""
    return DomainEvent(
        "conflict.resolved",
        {"conflict_type": conflict_type, "kept": kept, "invalidated": invalidated},
    )


__all__ = [
    "DomainEvent",
    "document_ingested",
    "query_answered",
    "feedback_received",
    "agent_trained",
    "conflict_resolved",
]
