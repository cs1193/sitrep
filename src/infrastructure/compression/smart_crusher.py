"""SmartCrusher: structural JSON compression.

Flattens nested structures, strips redundant (null/empty) keys, applies
type-aware encoding (booleans → 0/1, nulls dropped, long strings truncated), and
preserves arrays with truncation. Emits compact JSON sized to the target token
*ratio*.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.utils.common import count_tokens_heuristic

_logger = logging.getLogger("sitrep.compression.smart_crusher")


def _clamp_ratio(ratio: float) -> float:
    """Clamp *ratio* to [0.05, 1.0]."""
    return max(0.05, min(1.0, float(ratio)))


class SmartCrusher:
    """Lossy-but-structural JSON compressor."""

    def __init__(self, max_array: int = 8, max_string: int = 120, flatten_below: float = 0.5) -> None:
        """Configure array/string caps and the ratio threshold for flattening."""
        self.max_array = max_array
        self.max_string = max_string
        self.flatten_below = flatten_below

    # ----------------------------------------------------------------- public
    def compress(self, text: str, ratio: float = 0.5, **_: Any) -> str:
        """Compress JSON *text* toward *ratio* of its token budget; return a string."""
        ratio = _clamp_ratio(ratio)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            _logger.debug("SmartCrusher: input is not valid JSON; returning as-is")
            return text
        if ratio < self.flatten_below and isinstance(data, (dict, list)):
            payload: Any = self._flatten(data)
        else:
            payload = self._transform(data)
        compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return self._to_budget(compact, text, ratio)

    # ----------------------------------------------------------------- transforms
    def _transform(self, node: Any) -> Any:
        """Recursively strip redundant keys and type-encode values."""
        if isinstance(node, dict):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if self._is_redundant(value):
                    continue
                out[str(key)] = self._transform(value)
            return out
        if isinstance(node, list):
            return self._encode_list([self._transform(v) for v in node if not self._is_redundant(v)])
        return self._encode_scalar(node)

    def _flatten(self, node: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested objects to dot-notation keys."""
        flat: Dict[str, Any] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                if self._is_redundant(value):
                    continue
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    flat.update(self._flatten(value, path))
                else:
                    flat[path] = self._encode_list(value) if isinstance(value, list) else self._encode_scalar(value)
        elif isinstance(node, list):
            flat["#"] = self._encode_list(node)
        else:
            flat["#"] = self._encode_scalar(node)
        return flat

    # ----------------------------------------------------------------- encoders
    def _encode_scalar(self, value: Any) -> Any:
        """Type-aware scalar encoding."""
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            return value if len(value) <= self.max_string else value[: self.max_string] + "…"
        return value

    def _encode_list(self, items: Sequence[Any]) -> Any:
        """Preserve arrays, truncating to ``max_array`` with a tail marker."""
        cleaned = [self._encode_scalar(i) if not isinstance(i, (dict, list)) else self._transform(i) for i in items]
        if len(cleaned) <= self.max_array:
            return cleaned
        head = cleaned[: self.max_array]
        return head + [f"…(+{len(cleaned) - self.max_array} more)"]

    @staticmethod
    def _is_redundant(value: Any) -> bool:
        """Return True for empty containers, None, and empty strings."""
        if value is None:
            return True
        if isinstance(value, str) and value == "":
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    # ----------------------------------------------------------------- budget
    def _to_budget(self, compact: str, original: str, ratio: float) -> str:
        """Truncate the compact string to *ratio* of the original token count."""
        full_tokens = max(1, count_tokens_heuristic(original))
        target = max(1, int(full_tokens * ratio))
        current = count_tokens_heuristic(compact)
        if current <= target:
            return compact
        # Char-level budget preserving as much structure as possible.
        char_target = max(16, int(len(compact) * target / max(1, current)))
        _logger.debug("SmartCrusher truncating JSON %d→%d tokens", current, target)
        return compact[:char_target] + "…"
