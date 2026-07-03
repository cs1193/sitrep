"""Lightweight embedding utilities.

Provides a deterministic, dependency-free fallback embedder so the system can
operate without ``sentence-transformers``. Real dense embeddings are produced by
``src.infrastructure.embedding.sentence_transformer`` (extra ``[rag]``) when
available.
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Sequence

from src.utils.constants import EMBEDDING_DIM


def _signed_hash(token: str, salt: str = "") -> int:
    """Return a signed 64-bit hash of *token* (+ optional *salt*)."""
    h = hashlib.md5(f"{salt}{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=True)


def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deterministic feature-hashing embedding.

    Tokens are projected (bag-of-tokens) into a *dim*-dimensional vector with
    signed contributions, then L2-normalized. Not semantically rich, but stable
    and useful as a zero-dependency fallback / smoke-test embedder.
    """
    vec = [0.0] * dim
    for tok in (text or "").lower().split():
        if not tok:
            continue
        idx = abs(_signed_hash(tok)) % dim
        sign = 1.0 if (_signed_hash(tok, salt="sign") % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def batch_hash_embeddings(texts: Sequence[str], dim: int = EMBEDDING_DIM) -> List[List[float]]:
    """Embed a batch of texts with :func:`hash_embedding`."""
    return [hash_embedding(t, dim) for t in texts]


def dim_of(model_dim: int = EMBEDDING_DIM) -> int:
    """Return the active embedding dimensionality (helper for DI)."""
    return int(model_dim)
