"""Append-only JSONL event log (WAL) for audit + replay/recovery.

Subscribes to the in-process :class:`~src.infrastructure.event_bus.EventBus` to
capture domain events, and supports ``replay(apply_fn)`` to re-apply them — the
foundation for recovering a truncated store to consistency.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.utils.common import utc_now_iso

_logger = logging.getLogger("sitrep.persistence.event_log")

_DEFAULT_TOPICS = (
    "document.ingested",
    "query.answered",
    "feedback.received",
    "agent.trained",
    "conflict.resolved",
)


class EventLog:
    """Thread-safe append-only JSONL log of events."""

    def __init__(self, path) -> None:
        """Create the log at *path* (parent dirs created as needed)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: Dict[str, Any]) -> None:
        """Append one event as a JSONL line."""
        if "timestamp" not in event:
            event = {**event, "timestamp": utc_now_iso()}
        line = json.dumps(event, default=str, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        """Return all logged events in order (skips unparseable lines)."""
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # pragma: no cover
                    _logger.warning("skipping unparseable event-log line")
        return out

    def replay(self, apply_fn: Callable[[Dict[str, Any]], None]) -> int:
        """Re-apply every event through *apply_fn*; return the count applied."""
        n = 0
        for event in self.read_all():
            try:
                apply_fn(event)
                n += 1
            except Exception as exc:  # pragma: no cover
                _logger.warning("event-log replay failed at #%d: %s", n, exc)
        return n

    def subscribe(self, bus, topics: Iterable[str] = _DEFAULT_TOPICS) -> None:
        """Subscribe the log to each topic on *bus* (audit capture)."""
        for topic in topics:

            def _handler(payload, _t=topic):
                self.append({"topic": _t, "payload": payload})

            bus.subscribe(topic, _handler)

    def clear(self) -> None:
        """Truncate the log (primarily for tests)."""
        with self._lock:
            self.path.write_text("", encoding="utf-8")
