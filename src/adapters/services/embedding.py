"""Embedding service adapter: caches and instruments the configured embedder."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from src.domain.interfaces import EmbeddingGateway

_logger = logging.getLogger("sitrep.services.embedding")


class EmbeddingService(EmbeddingGateway):
    """Caching wrapper around an :class:`EmbeddingGateway` with metrics."""

    def __init__(self, gateway: EmbeddingGateway, cache_limit: int = 4096) -> None:
        """Wrap *gateway* with an LRU-bounded text→vector cache."""
        self.gateway = gateway
        self.dim = getattr(gateway, "dim", 384)
        self.name = getattr(gateway, "name", "embedding-service")
        self._cache: Dict[str, List[float]] = {}
        self._cache_limit = cache_limit

    def embed(self, text: str) -> List[float]:
        """Embed *text*, returning a cached vector when available."""
        if text in self._cache:
            return self._cache[text]
        vec = self.gateway.embed(text)
        self._maybe_cache(text, vec)
        return vec

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch, computing only the uncached texts."""
        if not texts:
            return []
        results: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        missing_idx = [i for i, t in enumerate(texts) if t not in self._cache]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            computed = self.gateway.embed_batch(missing_texts)
            for i, vec in zip(missing_idx, computed):
                results[i] = vec
                self._maybe_cache(texts[i], vec)
        for i, t in enumerate(texts):
            if results[i] is None:
                results[i] = self._cache[t]
        return results

    def _maybe_cache(self, text: str, vec: List[float]) -> None:
        """Store *vec* for *text*, evicting the oldest entry if over the limit."""
        if len(self._cache) >= self._cache_limit:
            self._cache.pop(next(iter(self._cache), None), None)
        self._cache[text] = vec

    def clear_cache(self) -> None:
        """Drop all cached vectors."""
        self._cache.clear()
