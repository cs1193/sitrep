"""Intent classification + routing (Phase D1).

Rule-based by default (keyword/pattern), with an optional LLM/zero-shot path.
Routes a query to the right downstream handler (simple lookup, comparison,
multi-hop, temporal, causal, multimodal, aggregation).
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional

from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.query.intent")


class IntentType(str, Enum):
    """Supported query intents."""

    SIMPLE = "simple"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    MULTIMODAL = "multimodal"
    AGGREGATION = "aggregation"


# Keyword cues per intent (order matters — first match wins).
_CUES = [
    (IntentType.MULTIMODAL, ("image", "picture", "photo", "figure", "diagram")),
    (IntentType.COMPARISON, ("compare", " vs ", "versus", "difference between", "better than")),
    (IntentType.CAUSAL, ("why ", "cause of", "effect of", "what if", "because", "leads to", "due to")),
    (IntentType.TEMPORAL, ("when ", "before ", "after ", "during ", "currently", "timeline", "in 20")),
    (IntentType.AGGREGATION, ("how many", "count of", "list all", "average", "sum of", "total ")),
    (
        IntentType.MULTI_HOP,
        ("how does", "how is", "how are", "relate", "relationship", "connect", "linked to", "inferred"),
    ),
]
_MULTI_QUESTION_RE = re.compile(r"\?.*\?")


class IntentClassifier:
    """Classifies a query into an :class:`IntentType`."""

    def __init__(self, llm: Optional[LLMGateway] = None) -> None:
        """Store an optional LLM for richer classification."""
        self.llm = llm

    def classify(self, query: str) -> IntentType:
        """Return the detected intent (default SIMPLE)."""
        if not query:
            return IntentType.SIMPLE
        q = query.lower()
        for intent, cues in _CUES:
            if any(cue in q for cue in cues):
                return intent
        if _MULTI_QUESTION_RE.search(query) or query.count("?") > 1:
            return IntentType.MULTI_HOP
        return IntentType.SIMPLE
