"""Quality estimation: retrieval/answer relevance and coverage metrics."""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from src.domain.interfaces import EmbeddingGateway, Reranker
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.services.quality")


class QualityEstimator:
    """Computes answer-quality signals used for scoring and confidence."""

    def __init__(
        self,
        embedder: Optional[EmbeddingGateway] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        """Store optional embedder and reranker for relevance scoring."""
        self.embedder = embedder
        self.reranker = reranker

    def relevance(self, query: str, answer: str) -> float:
        """Return query↔answer relevance in [0, 1] (cross-encoder or cosine)."""
        if not query or not answer:
            return 0.0
        if self.reranker is not None:
            try:
                ranked = self.reranker.rerank(query, [answer])
                return float(ranked[0][1]) if ranked else 0.0
            except Exception:  # pragma: no cover
                pass
        if self.embedder is not None:
            return max(0.0, cosine_similarity(self.embedder.embed(query), self.embedder.embed(answer)))
        from src.utils.embedding import hash_embedding

        return max(0.0, cosine_similarity(hash_embedding(query), hash_embedding(answer)))

    @staticmethod
    def coverage(answer: str, passages: Sequence[str], top_terms: int = 8) -> float:
        """Return the fraction of *passages* whose key terms appear in *answer*."""
        if not passages or not answer:
            return 0.0
        answer_lower = answer.lower()
        hits = 0
        for passage in passages:
            terms = [t for t in passage.lower().split() if len(t) > 4]
            if not terms:
                continue
            distinctive = sorted(set(terms), key=lambda t: -len(t))[:top_terms]
            if any(term in answer_lower for term in distinctive):
                hits += 1
        return hits / len(passages)

    def overall(
        self,
        query: str,
        answer: str,
        passages: Sequence[str],
        confidence: float = 0.0,
    ) -> float:
        """Composite answer-quality score in [0, 1]."""
        rel = self.relevance(query, answer)
        cov = self.coverage(answer, passages)
        return round(0.5 * rel + 0.3 * cov + 0.2 * confidence, 4)
