"""Result cache + query-frequency tracker (Phase E4).

The result cache memoizes QueryOrchestrator results keyed by
``(query, top_k, corpus_version)``; ingests bump the version via the event bus,
so cached entries naturally miss after the corpus changes. The frequency tracker
counts queries/terms to inform caching and (query-aware) consolidation.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("sitrep.cache.result")


class ResultCache:
    """In-memory query-result cache with TTL + corpus-version invalidation."""

    def __init__(self, ttl: int = 3600) -> None:
        """Configure the TTL (seconds; 0 = no expiry)."""
        self.ttl = int(ttl)
        self._store: Dict[str, Tuple[Any, float]] = {}
        self.version = 0
        self._lock = threading.RLock()

    def _key(self, query: str, top_k: Optional[int]) -> str:
        """Return a cache key derived from query, top_k, and corpus version."""
        raw = f"{(query or '').strip().lower()}|{top_k}|{self.version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get(self, query: str, top_k: Optional[int]) -> Any:
        """Return a cached DTO or None (expired entries are evicted)."""
        with self._lock:
            key = self._key(query, top_k)
            item = self._store.get(key)
            if item is None:
                return None
            dto, ts = item
            if self.ttl > 0 and (time.time() - ts) > self.ttl:
                self._store.pop(key, None)
                return None
            return dto

    def put(self, query: str, top_k: Optional[int], dto: Any) -> None:
        """Store *dto* under the (query, top_k) key."""
        with self._lock:
            self._store[self._key(query, top_k)] = (dto, time.time())

    def bump_version(self) -> None:
        """Invalidate the whole cache (called on corpus change)."""
        with self._lock:
            self.version += 1
            self._store.clear()
        _logger.debug("result cache invalidated (version=%d)", self.version)

    def subscribe(self, bus) -> None:
        """Invalidate on ``document.ingested`` events (decoupled from ingest)."""
        bus.subscribe("document.ingested", lambda _payload: self.bump_version())

    def __len__(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._store)


class QueryFrequencyTracker:
    """Counts queries and query terms (for caching + query-aware consolidation)."""

    def __init__(self) -> None:
        """Initialize empty counters."""
        self._queries: Dict[str, int] = {}
        self._terms: Dict[str, int] = {}

    def track(self, query: str) -> None:
        """Record one observation of *query*."""
        q = (query or "").strip()
        if not q:
            return
        self._queries[q] = self._queries.get(q, 0) + 1
        for term in q.lower().split():
            if len(term) > 3:
                self._terms[term] = self._terms.get(term, 0) + 1

    def count(self, query: str) -> int:
        """Return how often *query* has been seen."""
        return self._queries.get((query or "").strip(), 0)

    def hot_queries(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the *n* most frequent queries."""
        return sorted(self._queries.items(), key=lambda x: -x[1])[:n]

    def top_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the *n* most frequent query terms."""
        return sorted(self._terms.items(), key=lambda x: -x[1])[:n]
