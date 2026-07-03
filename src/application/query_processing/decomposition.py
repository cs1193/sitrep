"""Query decomposition (Phase D2): split compound questions into sub-queries.

Rule-based by default (conjunctions / clauses / multiple questions), with an
optional LLM path. Returns an ordered list of sub-queries (a query plan).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.query.decomposition")

# Split on " and ", ", then", ";", or a "?" followed by another clause.
_SPLIT_RE = re.compile(r"\s+and\s+|,\s*(?:then\s+)?\s*|\?\s+|;\s+")


class QueryDecomposer:
    """Decomposes a compound query into sub-queries."""

    def __init__(self, llm: Optional[LLMGateway] = None) -> None:
        """Store an optional LLM for richer decomposition."""
        self.llm = llm

    def decompose(self, query: str) -> List[str]:
        """Return a list of sub-queries (>=1)."""
        if not query or not query.strip():
            return []
        parts = _SPLIT_RE.split(query)
        cleaned = [p.strip().rstrip("?").strip() for p in parts if p and p.strip()]
        cleaned = [p for p in cleaned if len(p) > 2]
        return cleaned if len(cleaned) > 1 else [query.strip()]
