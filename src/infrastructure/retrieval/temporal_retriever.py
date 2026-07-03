"""Temporal retriever: point-in-time fact queries (Graphiti-style)."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from src.domain.interfaces import EmbeddingGateway, FactRepository
from src.domain.schemas import Fact
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.retrieval.temporal")


class TemporalRetriever:
    """Rank facts that were valid at a given *moment* by query relevance."""

    def __init__(
        self,
        fact_repo: FactRepository,
        embedder: Optional[EmbeddingGateway] = None,
    ) -> None:
        """Wire the fact repository and optional embedder for relevance ranking."""
        self.fact_repo = fact_repo
        self.embedder = embedder

    def retrieve_at(self, query: str, moment: datetime, top_k: int = 10) -> List[Fact]:
        """Return facts valid at *moment*, ranked by relevance to *query*."""
        try:
            facts = self.fact_repo.point_in_time(moment)
        except Exception as exc:  # pragma: no cover
            _logger.warning("point_in_time failed: %s", exc)
            return []
        if not facts:
            return []

        if self.embedder is not None:
            qv = self.embedder.embed(query)
            scored = [
                (cosine_similarity(qv, self.embedder.embed(f.triple)), f) for f in facts
            ]
        else:
            qterms = {t.lower() for t in query.split()}
            scored = []
            for f in facts:
                text = (f.subject + " " + f.predicate + " " + f.object_value).lower()
                overlap = sum(1 for t in qterms if t in text)
                scored.append((float(overlap), f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]
