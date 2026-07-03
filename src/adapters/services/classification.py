"""Document classification: zero-shot domain tagging + importance scoring.

Default domain classifier uses a keyword lexicon (no model). When
``transformers`` is available, a true zero-shot classifier can be wired in via
``set_zeroshot``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from src.domain.value_objects import Domain
from src.utils.common import count_tokens_heuristic

_logger = logging.getLogger("sitrep.services.classification")

_DOMAIN_LEXICON: Dict[str, Tuple[str, ...]] = {
    Domain.MEDICAL.value: (
        "patient", "diagnosis", "treatment", "dose", "symptom", "clinical", "disease",
        "therapy", "drug", "prescription", "medical", "health",
    ),
    Domain.TECHNICAL.value: (
        "api", "server", "code", "function", "algorithm", "database", "deploy", "software",
        "system", "module", "endpoint", "request", "config", "runtime",
    ),
    Domain.LEGAL.value: (
        "court", "statute", "contract", "clause", "liability", "jurisdiction", "plaintiff",
        "defendant", "legal", "law", "regulation", "compliance",
    ),
    Domain.FINANCIAL.value: (
        "revenue", "expense", "tax", "asset", "liability", "equity", "invoice", "payment",
        "budget", "financial", "profit", "fiscal",
    ),
    Domain.SCIENTIFIC.value: (
        "study", "hypothesis", "experiment", "data", "analysis", "method", "result",
        "research", "theory", "observation", "sample", "measurement",
    ),
}


class ClassificationService:
    """Domain + importance classifier with optional zero-shot backend."""

    def __init__(self, zeroshot: Optional[Any] = None) -> None:
        """Optionally accept a HuggingFace zero-shot classification pipeline."""
        self._zeroshot = zeroshot

    def set_zeroshot(self, pipeline: Any) -> None:
        """Inject a transformers zero-shot classification pipeline."""
        self._zeroshot = pipeline

    def classify(self, text: str) -> Tuple[str, float]:
        """Return ``(domain, confidence)`` for *text*."""
        if not text or not text.strip():
            return (Domain.GENERAL.value, 0.0)
        if self._zeroshot is not None:
            try:
                labels = [d.value for d in Domain]
                result = self._zeroshot(text[:2000], labels, multi_label=False)
                idx = max(range(len(result["labels"])), key=lambda i: result["scores"][i])
                return (result["labels"][idx], float(result["scores"][idx]))
            except Exception as exc:  # pragma: no cover
                _logger.debug("zeroshot failed, using lexicon: %s", exc)
        return self._lexicon_classify(text)

    @staticmethod
    def _lexicon_classify(text: str) -> Tuple[str, float]:
        """Score *text* against the domain keyword lexicon."""
        lower = text.lower()
        scores: Dict[str, int] = {}
        for domain, words in _DOMAIN_LEXICON.items():
            scores[domain] = sum(1 for w in words if w in lower)
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return (Domain.GENERAL.value, 0.0)
        total = sum(scores.values())
        return (best, scores[best] / max(1.0, float(total)))

    @staticmethod
    def importance(text: str, created_at: Optional[datetime] = None) -> float:
        """Return an importance score in [0, 1] from length and recency."""
        tokens = count_tokens_heuristic(text)
        length_factor = min(1.0, tokens / 256.0)
        recency_factor = 1.0
        if created_at is not None:
            now = datetime.now(timezone.utc)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
            recency_factor = max(0.25, 1.0 / (1.0 + age_days / 30.0))
        return round(0.6 * length_factor + 0.4 * recency_factor, 4)
