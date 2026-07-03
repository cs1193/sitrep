"""ContentRouter + content-type detection.

Classifies input as JSON / code / log / natural-language and dispatches to the
matching compressor. ``compress`` returns ``(compressed_text, metadata)`` where
metadata records the detected type, compressor used, and token telemetry.
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from src.infrastructure.compression.code_compressor import CodeCompressor
from src.infrastructure.compression.kompress import Kompress
from src.infrastructure.compression.log_compressor import LogCompressor
from src.infrastructure.compression.smart_crusher import SmartCrusher
from src.utils.common import count_tokens_heuristic

_logger = logging.getLogger("sitrep.compression.router")


class ContentType(str, Enum):
    """Detected content categories."""

    JSON = "json"
    CODE = "code"
    LOG = "log"
    TEXT = "text"


_TS_HINT = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\d{2}:\d{2}:\d{2}")
_LOG_LEVEL = re.compile(r"\b(ERROR|ERR|WARN(ING)?|INFO|DEBUG|FATAL|CRITICAL|TRACE)\b")
_CODE_HINTS = re.compile(
    r"\b(def |class |import |from \w+ import|function |const |let |var |public |private |"
    r"#include|return |print\(|console\.log|=>|;\s*$)\b|;\s*$",
    re.MULTILINE,
)


class ContentRouter:
    """Routes text to a content-aware compressor."""

    def __init__(
        self,
        smart_crusher: Optional[SmartCrusher] = None,
        code_compressor: Optional[CodeCompressor] = None,
        log_compressor: Optional[LogCompressor] = None,
        kompress: Optional[Kompress] = None,
    ) -> None:
        """Construct with compressors (defaults are created lazily)."""
        self.smart_crusher = smart_crusher or SmartCrusher()
        self.code_compressor = code_compressor or CodeCompressor()
        self.log_compressor = log_compressor or LogCompressor()
        self.kompress = kompress

    # ----------------------------------------------------------------- detection
    def detect(self, text: str) -> ContentType:
        """Classify *text* into a :class:`ContentType`."""
        s = (text or "").strip()
        if not s:
            return ContentType.TEXT
        if s[0] in "{[":
            try:
                json.loads(s)
                return ContentType.JSON
            except (json.JSONDecodeError, TypeError):
                pass
        lines = s.splitlines()
        if len(lines) >= 3:
            sample = lines[:25]
            ts_hits = sum(1 for line in sample if _TS_HINT.search(line))
            level_hits = sum(1 for line in sample if _LOG_LEVEL.search(line))
            has_traceback = "Traceback" in s or bool(_LOG_LEVEL.search(s))
            if (ts_hits >= max(2, len(sample) // 3)) or (has_traceback and ts_hits >= 1) or (
                level_hits >= 3 and ts_hits >= 2
            ):
                return ContentType.LOG
        if self._code_score(s) >= 2:
            return ContentType.CODE
        return ContentType.TEXT

    @staticmethod
    def _code_score(text: str) -> int:
        """Return a coarse code-likelihood score."""
        score = 0
        if _CODE_HINTS.search(text):
            score += 2
        if re.search(r"[{};]", text) and text.count("\n") >= 2:
            score += 1
        if re.search(r"^\s*(if|for|while|try|catch|elif|else)\b", text, re.MULTILINE):
            score += 1
        return score

    # ----------------------------------------------------------------- dispatch
    def compress(
        self, text: str, ratio: float = 0.5, query: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress *text* via the type-matched compressor; return (text, metadata)."""
        ctype = self.detect(text)
        full_tokens = count_tokens_heuristic(text)
        try:
            if ctype == ContentType.JSON:
                out = self.smart_crusher.compress(text, ratio=ratio)
                compressor = "smart_crusher"
            elif ctype == ContentType.CODE:
                out = self.code_compressor.compress(text, ratio=ratio)
                compressor = "code_compressor"
            elif ctype == ContentType.LOG:
                out = self.log_compressor.compress(text, ratio=ratio)
                compressor = "log_compressor"
            else:
                if self.kompress is None:
                    # No NL compressor wired → fall back to a head/tail truncation.
                    out = _fallback_truncate(text, ratio)
                    compressor = "truncate_fallback"
                else:
                    out = self.kompress.compress(text, ratio=ratio, query=query)
                    compressor = "kompress"
        except Exception as exc:  # pragma: no cover
            _logger.warning("router compressor %s failed (%s); using fallback", ctype.value, exc)
            out = _fallback_truncate(text, ratio)
            compressor = "fallback"
        compressed_tokens = count_tokens_heuristic(out)
        metadata = {
            "content_type": ctype.value,
            "compressor": compressor,
            "original_tokens": full_tokens,
            "compressed_tokens": compressed_tokens,
        }
        _logger.debug("router: %s via %s (%d→%d tokens)", ctype.value, compressor, full_tokens, compressed_tokens)
        return out, metadata


def _fallback_truncate(text: str, ratio: float) -> str:
    """Head/tail truncation used when no NL compressor is wired."""
    ratio = max(0.05, min(1.0, float(ratio)))
    if not text:
        return text
    lines = text.splitlines()
    n = max(2, int(12 * ratio))
    if len(lines) <= 2 * n:
        return text
    return "\n".join(lines[:n] + [f"… ({len(lines) - 2 * n} lines omitted) …"] + lines[-n:])
