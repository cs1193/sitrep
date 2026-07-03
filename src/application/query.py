"""Knowledge query use case (Headroom-enabled).

Pipeline: hybrid retrieval → RL-selected ratio → content-aware compression
(:class:`ContentRouter`) → reversible CCR storage → cache-aligned answer
generation (:class:`CacheAligner`) → multi-signal confidence + quality → active
learning. When Headroom components are absent it transparently falls back to the
original extractive :class:`CompressionService`, preserving backward
compatibility.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from src.adapters.repositories.ccr_repo import SQLiteCCRRepository
from src.adapters.services.compression import CompressionService
from src.adapters.services.confidence import ConfidenceEstimator
from src.adapters.services.quality import QualityEstimator
from src.application.active_learning import ActiveLearningService
from src.application.dto import QueryResultDTO
from src.application.events import query_answered
from src.domain.interfaces import CompressionPolicy, LLMGateway, Retriever
from src.domain.schemas import Decision
from src.infrastructure.cache import CacheAligner
from src.infrastructure.compression.router import ContentRouter
from src.infrastructure.db.sqlite_client import SQLiteClient
from src.infrastructure.lineage import LineageTracker
from src.infrastructure.monitoring.metrics import get_metrics
from src.utils.common import count_tokens_heuristic, generate_id, utc_now_iso
from src.utils.constants import DEC_COMPRESS, DEC_QUERY
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.usecase.query")


class QueryKnowledgeUseCase:
    """Answer a question over the local knowledge base (Headroom pipeline)."""

    def __init__(
        self,
        retriever: Retriever,
        compression_service: CompressionService,
        compression_policy: CompressionPolicy,
        llm: LLMGateway,
        confidence_estimator: ConfidenceEstimator,
        quality_estimator: QualityEstimator,
        active_learning: ActiveLearningService,
        lineage_tracker: LineageTracker,
        sqlite: SQLiteClient,
        config: Optional[Any] = None,
        presenter: Optional[Any] = None,
        content_router: Optional[ContentRouter] = None,
        cache_aligner: Optional[CacheAligner] = None,
        ccr_repo: Optional[SQLiteCCRRepository] = None,
    ) -> None:
        """Wire retrieval, compression, Headroom components, LLM, and telemetry."""
        self.retriever = retriever
        self.compression_service = compression_service
        self.compression_policy = compression_policy
        self.llm = llm
        self.confidence_estimator = confidence_estimator
        self.quality_estimator = quality_estimator
        self.active_learning = active_learning
        self.lineage_tracker = lineage_tracker
        self.sqlite = sqlite
        self.config = config
        self.presenter = presenter
        # Headroom components (all optional → graceful fallback).
        self.content_router = content_router
        self.cache_aligner = cache_aligner
        self.ccr_repo = ccr_repo

    @log_execution
    def execute(self, query: str, top_k: Optional[int] = None) -> QueryResultDTO:
        """Run the full Headroom query pipeline and return a :class:`QueryResultDTO`."""
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        query_id = generate_id("query")
        top_k = top_k or getattr(self.config, "top_k", 5)

        results = self.retriever.retrieve(query, top_k=top_k)
        context = "\n\n".join(r.text for r in results)

        # Observation → RL-selected compression ratio.
        observation = self._build_observation(query, results, context, top_k)
        ratio = self.compression_policy.select_ratio(observation)

        # Content-aware compression (Headroom) with extractive fallback.
        compressed, full_tokens, compressed_tokens, content_type, compressor = self._compress_context(
            context, ratio, query
        )

        # Reversible compression: retain the original locally for later retrieval.
        ccr_key = self._store_ccr(context, compressed, content_type, query, ratio, compressor)

        # Cache-aligned answer generation (stable system prefix for KV reuse).
        answer, cache_eligible = self._generate_answer(query, compressed)

        graph_density = self._graph_density()
        confidence = self.confidence_estimator.estimate(
            results, compression_ratio=ratio, graph_density=graph_density, top_k=top_k
        )
        quality = self.quality_estimator.overall(
            query, answer, [r.text for r in results], confidence
        )

        needs_clarification = self.active_learning.needs_clarification(confidence)
        clarification = self.active_learning.ask(query, results) if needs_clarification else None

        get_metrics().record_context(full_tokens, compressed_tokens)
        self._record_retrieval_stats(query_id, query, results)

        self.lineage_tracker.record(
            Decision(
                agent_id="query",
                decision_type=DEC_QUERY,
                action="answer_query",
                inputs={"query": query, "top_k": top_k},
                outputs={
                    "results": len(results),
                    "confidence": confidence,
                    "compression_ratio": ratio,
                    "content_type": content_type,
                    "compressor": compressor,
                    "ccr_key": ccr_key,
                    "cache_eligible": cache_eligible,
                    "needs_clarification": needs_clarification,
                },
                rationale=f"backend={getattr(self.llm, 'name', '?')}; quality={quality:.3f}",
            )
        )
        self.lineage_tracker.record(
            Decision(
                agent_id="headroom",
                decision_type=DEC_COMPRESS,
                action="compress_context",
                inputs={"ratio": ratio, "content_type": content_type, "compressor": compressor},
                outputs={"full_tokens": full_tokens, "compressed_tokens": compressed_tokens},
                rationale="content-aware compression",
            )
        )
        dto = QueryResultDTO(
            query_id=query_id,
            query=query,
            answer=answer,
            results=results,
            confidence=confidence,
            quality=quality,
            compression_ratio=ratio,
            full_tokens=full_tokens,
            compressed_tokens=compressed_tokens,
            needs_clarification=needs_clarification,
            clarification_question=clarification,
            backend=getattr(self.llm, "name", "demo"),
            content_type=content_type,
            compressor=compressor,
            ccr_key=ccr_key,
            cache_eligible=cache_eligible,
        )
        query_answered(query_id, query, confidence, dto.token_reduction).publish()
        return dto

    # ----------------------------------------------------------------- Headroom helpers
    def _compress_context(
        self, context: str, ratio: float, query: str
    ) -> Tuple[str, int, int, str, str]:
        """Compress *context*; prefer the content router, else the extractive service.

        Returns ``(compressed, original_tokens, compressed_tokens, content_type, compressor)``.
        """
        if not context:
            return ("", 0, 0, "text", "none")
        if self.content_router is not None:
            try:
                compressed, meta = self.content_router.compress(context, ratio=ratio, query=query)
                return (
                    compressed,
                    int(meta.get("original_tokens", count_tokens_heuristic(context))),
                    int(meta.get("compressed_tokens", count_tokens_heuristic(compressed))),
                    str(meta.get("content_type", "text")),
                    str(meta.get("compressor", "router")),
                )
            except Exception as exc:  # pragma: no cover
                _logger.warning("content router failed (%s); using extractive fallback", exc)
        compressed, full_tokens, compressed_tokens = self.compression_service.compress(
            context, ratio=ratio, query=query
        )
        return compressed, full_tokens, compressed_tokens, "text", "extractive"

    def _store_ccr(
        self, context: str, compressed: str, content_type: str, query: str, ratio: float, compressor: str
    ) -> Optional[str]:
        """Store the original context for reversible retrieval; return its key."""
        if self.ccr_repo is None or not context:
            return None
        try:
            return self.ccr_repo.store(
                original=context,
                compressed=compressed,
                content_type=content_type,
                metadata={"query": query, "ratio": ratio, "compressor": compressor},
            )
        except Exception as exc:  # pragma: no cover
            _logger.warning("CCR store failed: %s", exc)
            return None

    # ----------------------------------------------------------------- answer generation
    def _generate_answer(self, query: str, context: str) -> Tuple[str, Optional[bool]]:
        """Generate an answer; return ``(answer, cache_eligible)``."""
        if not context:
            return ("I have no relevant information to answer that.", None)
        prompt = (
            f"Context:\n{context}\n\nQuestion: {query}\n"
            "Answer concisely using only the context above:"
        )
        system: Optional[str] = None
        cache_eligible: Optional[bool] = None
        prompt_to_send = prompt
        if self.cache_aligner is not None:
            system, prompt_to_send, align_meta = self.cache_aligner.align(prompt)
            cache_eligible = bool(align_meta.get("cache_eligible"))
        try:
            answer = self.llm.generate(prompt_to_send, system=system)
        except Exception as exc:  # pragma: no cover
            _logger.warning("answer generation failed: %s", exc)
            return (f"(answer generation failed: {exc})", cache_eligible)
        return answer, cache_eligible

    # ----------------------------------------------------------------- legacy helpers
    def _build_observation(self, query, results, context, top_k):
        """Build the compression-policy observation (embedding + 3 stats)."""
        embedder = getattr(self.retriever, "embedder", None)
        if embedder is not None:
            emb = list(embedder.embed(query))
        else:
            from src.utils.embedding import hash_embedding

            emb = list(hash_embedding(query))
        conf = max((r.score for r in results), default=0.0)
        n_ratio = len(results) / max(1, top_k)
        ctx_ratio = min(1.0, count_tokens_heuristic(context) / 2048.0)
        return emb + [float(conf), float(n_ratio), float(ctx_ratio)]

    def _graph_density(self) -> float:
        """Return graph density if a graph store is wired, else 0."""
        graph_store = getattr(self.retriever, "graph_store", None)
        if graph_store is None:
            return 0.0
        try:
            return float(graph_store.density())
        except Exception:
            return 0.0

    def _record_retrieval_stats(self, query_id: str, query: str, results) -> None:
        """Persist per-channel scores for later fusion-weight learning."""
        bm = [float(r.bm25_score) for r in results]
        vec = [float(r.vector_score) for r in results]
        graph = [float(r.graph_score) for r in results]
        with self.sqlite.transaction():
            self.sqlite.execute(
                "INSERT OR REPLACE INTO retrieval_stats "
                "(query_id, query_text, bm25_scores, vector_scores, graph_scores, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    query_id,
                    query,
                    self.sqlite.dumps_json(bm),
                    self.sqlite.dumps_json(vec),
                    self.sqlite.dumps_json(graph),
                    utc_now_iso(),
                ),
            )
