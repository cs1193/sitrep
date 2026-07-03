"""Lightweight metrics collector (counters, gauges, histograms, timers).

No external backend; snapshots are queryable in-memory and surfaced in the
Stats UI. Tracks token-reduction analytics central to SITREP's value prop.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

_logger = logging.getLogger("sitrep.metrics")


class MetricsCollector:
    """Thread-safe in-memory metrics store."""

    def __init__(self) -> None:
        """Initialize empty counters, gauges, and observation buffers."""
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._observations: Dict[str, List[float]] = {}
        # token reduction analytics
        self._full_context_tokens: int = 0
        self._compressed_context_tokens: int = 0

    # ----------------------------------------------------------------- primitives
    def inc(self, name: str, by: float = 1.0) -> float:
        """Increment counter *name* by *by* and return the new value."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + by
            return self._counters[name]

    def gauge(self, name: str, value: float) -> None:
        """Set gauge *name* to *value*."""
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        """Record an observation *value* for histogram *name* (keeps last 1024)."""
        with self._lock:
            buf = self._observations.setdefault(name, [])
            buf.append(float(value))
            if len(buf) > 1024:
                del buf[: len(buf) - 1024]

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """Context manager timing a block into histogram *name* (seconds)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start)

    # ----------------------------------------------------------------- token reduction
    def record_context(self, full_tokens: int, compressed_tokens: int) -> None:
        """Record a compression event's full vs. compressed token counts."""
        with self._lock:
            self._full_context_tokens += int(full_tokens)
            self._compressed_context_tokens += int(compressed_tokens)
        self.inc("compression.events")

    @property
    def token_reduction_ratio(self) -> float:
        """Return the cumulative token-reduction ratio (0..1)."""
        with self._lock:
            full = self._full_context_tokens
            if full <= 0:
                return 0.0
            return 1.0 - (self._compressed_context_tokens / full)

    @property
    def tokens_saved(self) -> int:
        """Return the cumulative number of tokens saved by compression."""
        with self._lock:
            return max(0, self._full_context_tokens - self._compressed_context_tokens)

    # ----------------------------------------------------------------- snapshot
    def summary(self, name: str) -> Dict[str, float]:
        """Return count/min/mean/max for observations of *name*."""
        with self._lock:
            buf = list(self._observations.get(name, []))
        if not buf:
            return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "count": float(len(buf)),
            "min": min(buf),
            "mean": sum(buf) / len(buf),
            "max": max(buf),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a full metrics snapshot (counters, gauges, token analytics)."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.summary(k) for k in self._observations},
                "token_reduction_ratio": self.token_reduction_ratio,
                "tokens_saved": self.tokens_saved,
                "full_context_tokens": self._full_context_tokens,
                "compressed_context_tokens": self._compressed_context_tokens,
            }

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._observations.clear()
            self._full_context_tokens = 0
            self._compressed_context_tokens = 0


# --------------------------------------------------------------------------- singleton
_METRICS: MetricsCollector = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Return the process-wide :class:`MetricsCollector`."""
    return _METRICS


def reset_metrics() -> MetricsCollector:
    """Reset global metrics (primarily for tests) and return it."""
    global _METRICS
    _METRICS = MetricsCollector()
    return _METRICS
