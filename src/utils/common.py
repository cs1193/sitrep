"""Generic, dependency-light helpers shared across layers."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Sequence, TypeVar

T = TypeVar("T")
_log = logging.getLogger("sitrep.common")

# ~1.3 tokens per whitespace/word token is a reasonable heuristic average.
_WORD_RE = re.compile(r"\b\w+\b|[^\w\s]", re.UNICODE)


# --------------------------------------------------------------------------- time / ids
def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier, optionally prefixed (e.g. ``fact_…``)."""
    u = uuid.uuid4().hex
    return f"{prefix}_{u}" if prefix else u


def hash_text(text: str, algo: str = "sha256") -> str:
    """Return the hex digest of *text* using *algo*."""
    return hashlib.new(algo, (text or "").encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- json / text
def safe_json_dumps(obj: Any) -> str:
    """JSON-encode *obj*, falling back to ``str(obj)`` for non-serializable values."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def truncate(text: str, max_chars: int = 2000) -> str:
    """Truncate *text* to *max_chars*, appending an ellipsis if cut."""
    if not text:
        return ""
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def count_tokens_heuristic(text: str) -> int:
    """Approximate token count (~1.3 tokens per word-ish token)."""
    if not text:
        return 0
    return max(1, int(len(_WORD_RE.findall(text)) * 1.3))


# --------------------------------------------------------------------------- vectors
def normalize(vec: Sequence[float]) -> List[float]:
    """L2-normalize a vector; uses NumPy when available, else pure Python."""
    try:
        import numpy as np  # type: ignore

        v = np.asarray(vec, dtype="float64")
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return v.astype(float).tolist()
        return (v / n).tolist()
    except Exception:  # pragma: no cover - numpy is a core dep but be safe
        s = float(sum(x * x for x in vec) ** 0.5) or 1.0
        return [float(x) / s for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors (0 if either is zero)."""
    try:
        import numpy as np  # type: ignore

        va, vb = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:  # pragma: no cover
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


# --------------------------------------------------------------------------- collections / fs
def chunked(items: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield successive lists of length *size* from *items*."""
    if size <= 0:
        raise ValueError("size must be a positive integer")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def coerce_bool(value: Any) -> bool:
    """Coerce common truthy strings/ints to ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
