"""CacheAligner: stabilize the system-prompt prefix to maximize cache hits.

Prefix caching (Anthropic, OpenAI, and local runtimes) gives a large read
discount when the *leading* tokens of a request are byte-identical across calls.
``CacheAligner`` normalizes a fixed system prompt and pairs it with the variable
user content, reporting cache eligibility so the caller can decide to use it.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Tuple

from src.utils.common import hash_text

_logger = logging.getLogger("sitrep.cache.aligner")

DEFAULT_SYSTEM_PROMPT = (
    "You are SITREP, a precise context-engineering assistant. "
    "Answer using only the provided context. Be concise, factual, and avoid speculation. "
    "If the answer is not present in the context, say so explicitly."
)


class CacheAligner:
    """Produces a stable system prefix + variable user content per request."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, min_prefix_chars: int = 64) -> None:
        """Normalize and store the default system prompt and eligibility threshold."""
        self.system_prompt = self._normalize(system_prompt)
        self.min_prefix_chars = min_prefix_chars

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse whitespace so the prefix is byte-stable regardless of formatting."""
        return re.sub(r"\s+", " ", (text or "").strip())

    def align(
        self, user_content: str, system: str = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Return ``(system_prompt, user_content, telemetry)`` for a request.

        The returned system prompt is stable across calls (enabling prefix-cache
        hits); telemetry reports its hash, length, and cache eligibility.
        """
        sys_prompt = self._normalize(system) if system else self.system_prompt
        eligible = len(sys_prompt) >= self.min_prefix_chars
        telemetry = {
            "prefix_hash": hash_text(sys_prompt)[:12],
            "prefix_chars": len(sys_prompt),
            "cache_eligible": eligible,
        }
        _logger.debug("cache aligner: prefix_hash=%s eligible=%s", telemetry["prefix_hash"], eligible)
        return sys_prompt, (user_content or ""), telemetry

    def is_stable(self, system_a: str, system_b: str) -> bool:
        """Return True if two system prompts normalize to the same bytes."""
        return self._normalize(system_a) == self._normalize(system_b)
