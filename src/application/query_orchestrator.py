"""Query orchestrator (Phase D): intent routing over the query pipeline.

Classifies a query and routes it: SIMPLE → the existing
:class:`QueryKnowledgeUseCase`; MULTI_HOP → :class:`MultiHopReasoner`;
COMPARISON → decompose + per-sub-query answers merged. Other intents fall back to
the simple path with the intent annotated. ``QueryKnowledgeUseCase`` itself is
untouched, so the eval and existing callers are unaffected.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.application.dto import QueryResultDTO
from src.application.query_processing.decomposition import QueryDecomposer
from src.application.query_processing.intent import IntentClassifier, IntentType
from src.application.query_processing.multi_hop import MultiHopReasoner

_logger = logging.getLogger("sitrep.query.orchestrator")


class QueryOrchestrator:
    """Intent-aware front door over :class:`QueryKnowledgeUseCase`."""

    def __init__(
        self,
        query_uc: Any,
        classifier: IntentClassifier,
        decomposer: QueryDecomposer,
        multi_hop: MultiHopReasoner,
        cache: Any = None,
        tracker: Any = None,
        explanation_service: Any = None,
    ) -> None:
        """Wire the simple query use case + the intelligence + Phase E components."""
        self.query_uc = query_uc
        self.classifier = classifier
        self.decomposer = decomposer
        self.multi_hop = multi_hop
        self.cache = cache
        self.tracker = tracker
        self.explanation_service = explanation_service

    def execute(self, query: str, top_k: Optional[int] = None) -> QueryResultDTO:
        """Track frequency, consult the cache, dispatch, explain, and cache."""
        if self.tracker is not None:
            self.tracker.track(query)
        if self.cache is not None:
            cached = self.cache.get(query, top_k)
            if cached is not None:
                cached.cached = True
                return cached
        dto = self._dispatch(query, top_k)
        if self.explanation_service is not None:
            try:
                dto.explanation = self.explanation_service.explain(dto)
            except Exception:  # pragma: no cover
                pass
        if self.cache is not None:
            self.cache.put(query, top_k, dto)
        return dto

    def _dispatch(self, query: str, top_k: Optional[int] = None) -> QueryResultDTO:
        """Classify *query* and route to the matching handler; return a DTO."""
        intent = self.classifier.classify(query)
        if intent == IntentType.SIMPLE:
            dto = self.query_uc.execute(query, top_k)
            dto.intent = IntentType.SIMPLE.value
            return dto
        if intent == IntentType.MULTI_HOP:
            return self._handle_multi_hop(query, top_k, intent)
        if intent == IntentType.COMPARISON:
            return self._handle_comparison(query, top_k, intent)
        # TEMPORAL/CAUSAL/AGGREGATION/MULTIMODAL → simple path, intent annotated.
        dto = self.query_uc.execute(query, top_k)
        dto.intent = intent.value
        return dto

    # ----------------------------------------------------------------- handlers
    def _handle_multi_hop(self, query: str, top_k: Optional[int], intent: IntentType) -> QueryResultDTO:
        """Run multi-hop reasoning and wrap the result in a DTO."""
        result = self.multi_hop.reason(query, top_k=top_k or 5)
        results = list(result.get("results", []))
        chain = result.get("chain", [])
        answer = result.get("answer", "")
        dto = QueryResultDTO(
            query_id=f"mh_{abs(hash(query)) % (10**8)}",
            query=query,
            answer=answer,
            results=results,
            confidence=float(result.get("confidence", 0.5)),
            backend=getattr(self.query_uc.llm, "name", "demo"),
            intent=intent.value,
            extras={"hop_chain": chain, "entities": result.get("entities", [])},
        )
        return dto

    def _handle_comparison(self, query: str, top_k: Optional[int], intent: IntentType) -> QueryResultDTO:
        """Decompose the comparison, answer each side, and merge."""
        sub_queries = self.decomposer.decompose(query)
        sub_dtos = [self.query_uc.execute(sq, top_k) for sq in sub_queries]
        merged_answer = "\n\n".join(f"• {sq}:\n{d.answer}" for sq, d in zip(sub_queries, sub_dtos) if d.answer)
        # Merge + dedupe passage results.
        seen, results = set(), []
        for d in sub_dtos:
            for r in d.results:
                if r.passage_id not in seen:
                    seen.add(r.passage_id)
                    results.append(r)
        confidence = max((d.confidence for d in sub_dtos), default=0.0)
        return QueryResultDTO(
            query_id=f"cmp_{abs(hash(query)) % (10**8)}",
            query=query,
            answer=merged_answer or "(no comparable answers found)",
            results=results,
            confidence=confidence,
            backend=getattr(self.query_uc.llm, "name", "demo"),
            intent=intent.value,
        )
