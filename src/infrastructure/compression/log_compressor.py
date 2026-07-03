"""LogCompressor: log/trace compression.

Removes timestamps (ISO-8601 and common formats) and stack traces, deduplicates
consecutive identical lines, and keeps a head/tail window scaled by *ratio*.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List

_logger = logging.getLogger("sitrep.compression.log")

# Timestamps: ISO-8601, syslog-ish, bracketed Apache, and bare HH:MM:SS.
_TS_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\[\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}(?::\d{2})?\s+[+-]\d{4}\]"
    r"|\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r")\b"
)
_TRACEBACK_START = "Traceback"
_JAVA_FRAME = re.compile(r"^\s+at\s+")
_CAUSED_BY = re.compile(r"^\s*Caused by:")
_PY_FRAME = re.compile(r"^\s+(File\s|During handling|Traceback)")


def _clamp_ratio(ratio: float) -> float:
    """Clamp *ratio* to [0.05, 1.0]."""
    return max(0.05, min(1.0, float(ratio)))


class LogCompressor:
    """Timestamp/trace-stripping, deduplicating, head/tail log compressor."""

    def __init__(self, first_n: int = 12, last_m: int = 12) -> None:
        """Configure the head/tail window sizes (scaled by ratio at compress time)."""
        self.first_n = first_n
        self.last_m = last_m

    # ----------------------------------------------------------------- public
    def compress(self, text: str, ratio: float = 0.5, **_: Any) -> str:
        """Compress log *text* toward *ratio*; returns a cleaned, windowed string."""
        ratio = _clamp_ratio(ratio)
        lines = text.splitlines()
        lines = [self._strip_timestamp(line) for line in lines]
        lines = self._strip_traces(lines)
        lines = self._dedupe_consecutive(lines)
        windowed = self._head_tail(lines, ratio)
        return "\n".join(windowed)

    # ----------------------------------------------------------------- stages
    @staticmethod
    def _strip_timestamp(line: str) -> str:
        """Remove the first timestamp occurrence from *line*."""
        return _TS_RE.sub("", line, count=1).strip()

    @staticmethod
    def _strip_traces(lines: List[str]) -> List[str]:
        """Drop Python/Java stack-trace frames and traceback headers."""
        out: List[str] = []
        in_trace = False
        for line in lines:
            if _TRACEBACK_START in line:
                in_trace = True
                continue
            if in_trace:
                if _PY_FRAME.match(line) or _JAVA_FRAME.match(line) or _CAUSED_BY.match(line):
                    continue
                in_trace = False
            if _JAVA_FRAME.match(line) or _CAUSED_BY.match(line):
                continue
            out.append(line)
        return out

    @staticmethod
    def _dedupe_consecutive(lines: List[str]) -> List[str]:
        """Collapse runs of identical consecutive lines, annotating the count."""
        out: List[str] = []
        run = 1
        for line in lines:
            if out and out[-1].split("×")[-1].strip() == line and line:
                run += 1
                out[-1] = f"{line} ×{run}"
            else:
                run = 1
                out.append(line)
        return out

    def _head_tail(self, lines: List[str], ratio: float) -> List[str]:
        """Keep the first N and last M lines, scaling N/M by *ratio*."""
        n = max(2, int(self.first_n * ratio))
        m = max(2, int(self.last_m * ratio))
        if len(lines) <= n + m:
            return lines
        head = lines[:n]
        tail = lines[-m:]
        omitted = len(lines) - n - m
        return head + [f"… ({omitted} lines omitted) …"] + tail
