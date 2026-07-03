"""Kompress: natural-language compression.

Thin Headroom adapter that delegates to the existing extractive
:class:`~src.adapters.services.compression.CompressionService` (the "RTK-style"
text compressor). Kept in infrastructure with no adapter import — the concrete
compressor is injected via ``delegate`` to preserve dependency direction.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

_logger = logging.getLogger("sitrep.compression.kompress")


@runtime_checkable
class _TextCompressor(Protocol):
    """Structural type for any object with a ``compress(text, ratio, query)`` method."""

    def compress(self, text: str, ratio: float = 0.5, query: Optional[str] = None) -> Any: ...


class Kompress:
    """Natural-language compressor wrapping an injected text compressor."""

    name = "kompress"

    def __init__(self, delegate: Any) -> None:
        """Store the delegate (e.g. :class:`CompressionService`)."""
        self.delegate = delegate

    def compress(self, text: str, ratio: float = 0.5, query: Optional[str] = None, **_: Any) -> str:
        """Delegate to the wrapped compressor, returning the compressed text only."""
        if self.delegate is None:
            return text
        try:
            result = self.delegate.compress(text, ratio=ratio, query=query)
        except TypeError:
            # Delegate does not accept ``query``.
            result = self.delegate.compress(text, ratio=ratio)
        # CompressionService.compress returns (text, full_tokens, compressed_tokens).
        if isinstance(result, tuple) and result:
            return result[0]
        return str(result)
