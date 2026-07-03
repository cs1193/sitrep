"""CodeCompressor: AST-aware code compression.

Strips comments, docstrings, and insignificant whitespace while preserving
structure (Python via the ``ast`` module). Falls back to line-based comment /
whitespace removal when the input is not parseable Python (or AST is unavailable).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.utils.common import count_tokens_heuristic

_logger = logging.getLogger("sitrep.compression.code")


def _clamp_ratio(ratio: float) -> float:
    """Clamp *ratio* to [0.05, 1.0]."""
    return max(0.05, min(1.0, float(ratio)))


class CodeCompressor:
    """Structure-preserving source-code compressor."""

    def __init__(self, keep_docstrings_above: float = 0.6) -> None:
        """Set the ratio above which docstrings are retained."""
        self.keep_docstrings_above = keep_docstrings_above

    # ----------------------------------------------------------------- public
    def compress(self, text: str, ratio: float = 0.5, **_: Any) -> str:
        """Compress *text*; prefer AST, fall back to line-based removal."""
        ratio = _clamp_ratio(ratio)
        try:
            return self._ast_compress(text, ratio)
        except Exception as exc:
            _logger.debug("CodeCompressor AST path failed (%s); using line-based", exc)
            return self._line_compress(text, ratio)

    # ----------------------------------------------------------------- AST path
    def _ast_compress(self, text: str, ratio: float) -> str:
        """Round-trip through :mod:`ast` to drop comments/docstrings/whitespace."""
        import ast

        tree = ast.parse(text)
        if ratio < self.keep_docstrings_above:
            self._strip_docstrings(tree)
        return self._to_budget(ast.unparse(tree), text, ratio)

    @staticmethod
    def _strip_docstrings(tree: Any) -> None:
        """Remove the first statement if it is a docstring (string literal expression)."""
        import ast

        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())

    # ----------------------------------------------------------------- line path
    def _line_compress(self, text: str, ratio: float) -> str:
        """Remove comments (#, //), block-comment lines, and blank lines."""
        cleaned: list[str] = []
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if in_block:
                if "*/" in stripped:
                    in_block = False
                continue
            if stripped.startswith("/*"):
                if "*/" not in stripped:
                    in_block = True
                continue
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            line = re.sub(r"#.*$", "", line)
            line = re.sub(r"//.*$", "", line).rstrip()
            if line.strip():
                cleaned.append(line)
        return self._to_budget("\n".join(cleaned), text, ratio)

    # ----------------------------------------------------------------- budget
    @staticmethod
    def _to_budget(code: str, original: str, ratio: float) -> str:
        """Keep whole lines until the token budget is met (structure-preserving)."""
        full_tokens = max(1, count_tokens_heuristic(original))
        target = max(1, int(full_tokens * ratio))
        if count_tokens_heuristic(code) <= target:
            return code
        lines = code.splitlines()
        kept: list[str] = []
        used = 0
        for line in lines:
            tok = count_tokens_heuristic(line)
            if used + tok > target and kept:
                break
            kept.append(line)
            used += tok
        if len(kept) < len(lines):
            kept.append(f"# …(+{len(lines) - len(kept)} lines)")
        return "\n".join(kept)
