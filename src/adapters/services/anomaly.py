"""Anomaly detection (Phase F3): statistical outliers over memory signals.

Flags passages whose ``importance``, ``access_count``, or embedding
centroid-distance (novelty) are z-score outliers (|z| ≥ threshold). Pure Python,
no ML deps.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional

from src.domain.schemas import Passage
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.services.anomaly")


class AnomalyDetector:
    """Z-score anomaly detector over numeric memory signals."""

    def __init__(self, threshold: float = 2.5) -> None:
        """Set the |z| threshold above which a value is anomalous."""
        self.threshold = float(threshold)

    @staticmethod
    def _stats(values: List[float]) -> tuple:
        """Return (mean, std) of *values* (0,0 if empty)."""
        n = len(values)
        if n == 0:
            return (0.0, 0.0)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        return (mean, math.sqrt(var))

    def _z(self, value: float, mean: float, std: float) -> float:
        """Return the z-score (0 if no spread)."""
        return 0.0 if std == 0 else (value - mean) / std

    def detect(self, passages: Iterable[Passage]) -> List[Dict[str, Any]]:
        """Return a list of anomalies ``{passage_id, signal, value, z, severity}``."""
        passages = list(passages)
        if not passages:
            return []
        importance = [float(p.metadata.get("importance", 0.0)) for p in passages]
        access = [float(p.metadata.get("access_count", 0)) for p in passages]
        novelty = self._novelty(passages)
        imp_mean, imp_std = self._stats(importance)
        acc_mean, acc_std = self._stats(access)
        nov_mean, nov_std = self._stats(novelty)

        anomalies: List[Dict[str, Any]] = []
        for i, p in enumerate(passages):
            for signal, value, mean, std in (
                ("importance", importance[i], imp_mean, imp_std),
                ("access_count", access[i], acc_mean, acc_std),
                ("novelty", novelty[i], nov_mean, nov_std),
            ):
                z = self._z(value, mean, std)
                if abs(z) >= self.threshold:
                    anomalies.append({
                        "passage_id": p.id,
                        "signal": signal,
                        "value": round(value, 4),
                        "z": round(z, 3),
                        "severity": round(min(1.0, abs(z) / (2 * self.threshold)), 3),
                    })
        anomalies.sort(key=lambda a: abs(a["z"]), reverse=True)
        _logger.info("anomaly scan: %d anomalies over %d passages", len(anomalies), len(passages))
        return anomalies

    @staticmethod
    def _novelty(passages: List[Passage]) -> List[float]:
        """Return embedding centroid-distance (novelty) per passage (0 if no embedding)."""
        embs = [p.embedding for p in passages if p.embedding]
        if len(embs) < 2:
            return [0.0] * len(passages)
        dim = len(embs[0])
        centroid = [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
        out = []
        for p in passages:
            if p.embedding:
                out.append(max(0.0, 1.0 - cosine_similarity(p.embedding, centroid)))
            else:
                out.append(0.0)
        return out
