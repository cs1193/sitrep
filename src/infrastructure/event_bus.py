"""In-process event bus (synchronous pub/sub) for domain/application events.

Keeps layers decoupled: use cases publish events; cross-cutting concerns
(metrics, lineage) subscribe without being wired in directly. Thread-safe via a
single global lock around subscriber mutation.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

_logger = logging.getLogger("sitrep.events")

EventHandler = Callable[[Dict[str, Any]], None]


class EventBus:
    """A minimal synchronous event bus with topic-based subscriptions."""

    def __init__(self) -> None:
        """Initialize an empty bus."""
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register *handler* for events published to *topic*."""
        with self._lock:
            self._subscribers[topic].append(handler)
        _logger.debug("subscribed handler to '%s'", topic)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Dispatch *payload* to all subscribers of *topic* (failures are isolated)."""
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:  # pragma: no cover - never let one handler break others
                _logger.exception("event handler failed for topic '%s'", topic)

    def subscribers(self, topic: str) -> List[EventHandler]:
        """Return the handlers currently subscribed to *topic* (snapshot)."""
        with self._lock:
            return list(self._subscribers.get(topic, []))

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscribers.clear()


# --------------------------------------------------------------------------- singleton
_BUS: EventBus = EventBus()


def get_event_bus() -> EventBus:
    """Return the process-wide :class:`EventBus` singleton."""
    return _BUS


def reset_event_bus() -> EventBus:
    """Reset the global bus (primarily for tests) and return it."""
    global _BUS
    _BUS = EventBus()
    return _BUS
