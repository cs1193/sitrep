"""Embedding gateway factory (auto-detect).

Returns a :class:`SentenceTransformerEmbedder` when ``sentence-transformers`` is
available; otherwise a deterministic :class:`HashEmbedder` fallback so the rest
of the system runs with zero model downloads.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.domain.interfaces import EmbeddingGateway
from src.utils.constants import EMBEDDING_DIM

_logger = logging.getLogger("sitrep.embedding")


class HashEmbedder(EmbeddingGateway):
    """Dependency-free embedder backed by feature hashing."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.name = "hash"

    def embed(self, text: str) -> list:
        from src.utils.embedding import hash_embedding

        return hash_embedding(text, self.dim)

    def embed_batch(self, texts) -> list:
        from src.utils.embedding import batch_hash_embeddings

        return batch_hash_embeddings(texts, self.dim)


def get_embedding_gateway(config: Optional[Any] = None) -> EmbeddingGateway:
    """Resolve an embedder: sentence-transformers if importable, else hash."""
    from src.utils.config import get_config

    cfg = config or get_config()
    try:
        from src.infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            model_name=cfg.embedding_model,
            dim=cfg.embedding_dim,
            batch_size=cfg.embedding_batch_size,
        )
        if embedder.is_available():
            _logger.info("embedding gateway: sentence-transformers (%s)", cfg.embedding_model)
            return embedder
    except Exception as exc:  # pragma: no cover
        _logger.debug("sentence-transformers unavailable: %s", exc)
    _logger.info("embedding gateway: hash fallback (dim=%d)", cfg.embedding_dim)
    return HashEmbedder(dim=cfg.embedding_dim)


__all__ = ["HashEmbedder", "SentenceTransformerEmbedder", "get_embedding_gateway"]
