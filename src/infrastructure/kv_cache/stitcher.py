"""Online KV-cache stitching.

Concatenates precomputed KV caches from selected passages along the sequence
dimension, yielding a single ``past_key_values`` usable for generation. Handles
both the new ``DynamicCache`` API and the legacy tuple-of-(key, value) format.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from src.domain.interfaces import KVCacheRepository

_logger = logging.getLogger("sitrep.kv.stitcher")


class KVStitcher:
    """Stitch per-passage KV caches into one combined cache."""

    def __init__(self, kv_repo: KVCacheRepository) -> None:
        """Wire the KV-cache repository."""
        self.kv_repo = kv_repo

    def is_available(self) -> bool:
        """Return True if ``torch`` is importable (needed to concatenate tensors)."""
        try:
            import torch  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def stitch(self, passage_ids: Sequence[str]) -> Optional[Any]:
        """Load and concatenate caches for *passage_ids*; return None if empty."""
        caches: List[Any] = []
        for pid in passage_ids:
            cache = self.kv_repo.get(pid)
            if cache is not None:
                caches.append(cache)
        if not caches:
            return None
        if len(caches) == 1:
            return caches[0]
        return self._concat(caches)

    def _concat(self, caches: Sequence[Any]) -> Optional[Any]:
        """Concatenate caches along the sequence dimension (format-aware)."""
        if not self.is_available():
            _logger.warning("torch unavailable; cannot stitch caches")
            return None
        import torch  # type: ignore

        first = caches[0]
        # New DynamicCache API.
        if hasattr(first, "key_cache") and hasattr(first, "value_cache"):
            try:
                from transformers import DynamicCache  # type: ignore
            except ImportError:  # pragma: no cover
                return None
            merged = DynamicCache()
            n_layers = len(first.key_cache)
            for layer in range(n_layers):
                keys = torch.cat([c.key_cache[layer] for c in caches], dim=2)
                values = torch.cat([c.value_cache[layer] for c in caches], dim=2)
                merged.key_cache.append(keys)
                merged.value_cache.append(values)
            return merged

        # Legacy tuple/list of (key, value) per layer.
        if isinstance(first, (tuple, list)):
            merged = []
            for layer in range(len(first)):
                k = torch.cat([c[layer][0] for c in caches], dim=2)
                v = torch.cat([c[layer][1] for c in caches], dim=2)
                merged.append((k, v))
            return tuple(merged)

        _logger.warning("unknown KV cache format; returning first cache")
        return first
