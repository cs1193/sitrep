"""Multi-signal confidence estimation.

Combines retrieval similarity, graph density, compression factor, reranker
score, and result count into a single confidence value in [0, 1].
"""
from __future__ import annotations

import logging
from typing import Sequence

from src.domain.value_objects import RetrievalResult

_logger = logging.getLogger("sitrep.services.confidence")


class ConfidenceEstimator:
    """Weights several weak signals into a confidence score."""

    def __init__(
        self,
        w_retrieval: float = 0.40,
        w_rerank: float = 0.25,
        w_graph: float = 0.15,
        w_compression: float = 0.10,
        w_coverage: float = 0.10,
    ) -> None:
        """Configure per-signal weights (normalized to sum 1)."""
        total = w_retrieval + w_rerank + w_graph + w_compression + w_coverage
        self.w = {
            "retrieval": w_retrieval / total,
            "rerank": w_rerank / total,
            "graph": w_graph / total,
            "compression": w_compression / total,
            "coverage": w_coverage / total,
        }

    def estimate(
        self,
        results: Sequence[RetrievalResult],
        compression_ratio: float = 1.0,
        graph_density: float = 0.0,
        top_k: int = 5,
    ) -> float:
        """Return a confidence score in [0, 1] for a retrieval+compression result."""
        if not results:
            return 0.0
        top = results[0]
        retrieval = max(0.0, min(1.0, float(top.vector_score)))
        rerank = float(top.rerank_score) if top.rerank_score is not None else float(top.score)
        rerank = max(0.0, min(1.0, rerank))
        graph = max(0.0, min(1.0, float(graph_density)))
        # Less aggressive compression → higher confidence (more context retained).
        compression = max(0.0, min(1.0, float(compression_ratio)))
        coverage = min(1.0, len(results) / max(1, top_k))
        score = (
            self.w["retrieval"] * retrieval
            + self.w["rerank"] * rerank
            + self.w["graph"] * graph
            + self.w["compression"] * compression
            + self.w["coverage"] * coverage
        )
        return round(max(0.0, min(1.0, score)), 4)

    def is_confident(self, score: float, threshold: float = 0.55) -> bool:
        """Return True if *score* clears *threshold*."""
        return score >= threshold
