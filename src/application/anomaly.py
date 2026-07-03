"""Anomaly use case (Phase F3): scan memory for statistical anomalies."""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.adapters.services.anomaly import AnomalyDetector
from src.domain.interfaces import PassageRepository

_logger = logging.getLogger("sitrep.usecase.anomaly")


class AnomalyUseCase:
    """Scans passages for anomalous importance/access/novelty signals."""

    def __init__(self, passage_repo: PassageRepository, detector: AnomalyDetector) -> None:
        """Wire the passage repo + detector."""
        self.passage_repo = passage_repo
        self.detector = detector

    def execute(self) -> Dict[str, Any]:
        """Scan all passages and return ``{anomalies, n_scanned, by_signal}``."""
        passages = list(self.passage_repo.iter_all())
        # Embeddings make novelty meaningful; pull them where available.
        emb_by_id = {p.id: p.embedding for p in self.passage_repo.iter_with_embeddings()}
        for p in passages:
            if p.embedding is None and p.id in emb_by_id:
                p.embedding = emb_by_id[p.id]
        anomalies = self.detector.detect(passages)
        by_signal: Dict[str, int] = {}
        for a in anomalies:
            by_signal[a["signal"]] = by_signal.get(a["signal"], 0) + 1
        return {"n_scanned": len(passages), "anomalies": anomalies, "by_signal": by_signal}
