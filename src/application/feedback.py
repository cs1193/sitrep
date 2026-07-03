"""Feedback use case: record ratings and update fusion weights online."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.application.dto import FeedbackDTO
from src.application.events import feedback_received
from src.domain.interfaces import FeedbackRepository, Retriever
from src.domain.schemas import Decision
from src.infrastructure.db.sqlite_client import SQLiteClient
from src.infrastructure.lineage import LineageTracker
from src.utils.constants import DEC_FEEDBACK
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.usecase.feedback")

_POLARITY_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class FeedbackUseCase:
    """Records user feedback and nudges retriever fusion weights."""

    def __init__(
        self,
        feedback_repo: FeedbackRepository,
        retriever: Retriever,
        sqlite: SQLiteClient,
        lineage_tracker: LineageTracker,
        config: Optional[Any] = None,
    ) -> None:
        """Wire the feedback repository, retriever, sqlite, and lineage."""
        self.feedback_repo = feedback_repo
        self.retriever = retriever
        self.sqlite = sqlite
        self.lineage_tracker = lineage_tracker
        self.config = config

    @log_execution
    def submit(
        self,
        query_id: str,
        polarity: str,
        rating: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackDTO:
        """Record feedback, update fusion weights, and log lineage."""
        polarity = (polarity or "neutral").lower()
        feedback_id = self.feedback_repo.save(query_id, polarity, rating, metadata)

        weights_updated, new_weights = self._update_fusion(query_id, polarity)

        with self.sqlite.transaction():
            self.sqlite.execute(
                "UPDATE retrieval_stats SET feedback=? WHERE query_id=?", (float(rating), query_id)
            )

        self.lineage_tracker.record(
            Decision(
                agent_id="feedback",
                decision_type=DEC_FEEDBACK,
                action="record_feedback",
                inputs={"query_id": query_id, "polarity": polarity, "rating": rating},
                outputs={"weights_updated": weights_updated, "new_weights": list(new_weights or [])},
                rationale="online fusion-weight update",
            )
        )
        feedback_received(query_id, polarity, rating).publish()
        return FeedbackDTO(
            feedback_id=feedback_id,
            query_id=query_id,
            polarity=polarity,
            rating=float(rating),
            weights_updated=weights_updated,
            new_weights=new_weights,
        )

    def _update_fusion(self, query_id: str, polarity: str):
        """Push per-channel gradients from the stored retrieval stats."""
        row = self.sqlite.fetchone(
            "SELECT bm25_scores, vector_scores, graph_scores FROM retrieval_stats WHERE query_id=?",
            (query_id,),
        )
        if row is None:
            return False, None
        bm = self.sqlite.loads_json(row["bm25_scores"], [])
        vec = self.sqlite.loads_json(row["vector_scores"], [])
        graph = self.sqlite.loads_json(row["graph_scores"], [])
        if not (bm or vec or graph):
            return False, None
        sign = _POLARITY_SIGN.get(polarity, 0.0)
        n = max(len(bm), len(vec), len(graph))
        feedback_row = {
            str(i): (
                bm[i] if i < len(bm) else 0.0,
                vec[i] if i < len(vec) else 0.0,
                graph[i] if i < len(graph) else 0.0,
                sign,
            )
            for i in range(n)
        }
        if not feedback_row or sign == 0.0:
            return False, self.retriever.weights
        self.retriever.fusion.update_online([feedback_row])
        self.retriever.update_weights(self.retriever.weights)
        _logger.info("fusion weights updated via feedback: %s", self.retriever.weights)
        return True, self.retriever.weights
