"""Cross-encoder reranker (extra ``[rag]``) with a cosine-similarity fallback.

Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` when ``sentence-transformers`` is
installed; otherwise ranks by embedding cosine similarity so retrieval still
produces a meaningful rerank signal.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

from src.domain.interfaces import EmbeddingGateway, Reranker
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.retrieval.reranker")


class CrossEncoderReranker(Reranker):
    """Reranker backed by a HuggingFace cross-encoder (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedder: Optional[EmbeddingGateway] = None,
        device: str = "cpu",
    ) -> None:
        """Configure model name, fallback embedder, and device."""
        self.model_name = model_name
        self.embedder = embedder
        self.device = device
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True if ``sentence_transformers`` is importable."""
        try:
            import sentence_transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazily load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore

            _logger.info("loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)

    def _fallback_embed(self, text: str) -> Sequence[float]:
        """Embed *text* via the configured embedder or a hash fallback."""
        if self.embedder is not None:
            return self.embedder.embed(text)
        from src.utils.embedding import hash_embedding

        return hash_embedding(text)

    def rerank(self, query: str, documents: Sequence[str]) -> List[Tuple[int, float]]:
        """Return ``(document_index, score)`` pairs sorted by relevance desc.

        Scores are normalized to [0, 1] (sigmoid for the cross-encoder).
        """
        if not documents:
            return []
        if self.is_available():
            try:
                self._load()
                import numpy as np  # type: ignore

                raw = self._model.predict([(query, d) for d in documents])
                scores = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype="float64")))
                order = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
                return [(i, float(scores[i])) for i in order]
            except Exception as exc:  # pragma: no cover
                _logger.warning("cross-encoder rerank failed, using fallback: %s", exc)

        qv = self._fallback_embed(query)
        scored = [
            (i, max(0.0, cosine_similarity(qv, self._fallback_embed(d)))) for i, d in enumerate(documents)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
