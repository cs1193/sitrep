"""Sentence-Transformers embedder (extra ``[rag]``)."""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from src.domain.interfaces import EmbeddingGateway

_logger = logging.getLogger("sitrep.embedding.st")


class SentenceTransformerEmbedder(EmbeddingGateway):
    """Dense embeddings via ``sentence-transformers`` (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        """Store config; the model loads on first use."""
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self.name = "sentence-transformers"
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True if ``sentence_transformers`` is importable."""
        try:
            import sentence_transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazily load the underlying model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _logger.info("loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            try:
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:  # pragma: no cover
                pass

    def embed(self, text: str) -> List[float]:
        """Embed a single *text* to a normalized vector."""
        self._load()
        import numpy as np  # type: ignore

        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32").tolist()

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of *texts* (normalized)."""
        if not texts:
            return []
        self._load()
        import numpy as np  # type: ignore

        vecs = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32").tolist()
