"""Application layer: use cases, DTOs, events, and the composition root.

``build_application()`` is the dependency-injection root that wires every
adapter and infrastructure component into ready-to-use use cases. Scripts and
the Gradio UI consume the returned :class:`Application`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.application.dto import StatsDTO
from src.utils.config import SitrepConfig, get_config
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.application")


class Application:
    """Container holding wired use cases and shared components."""

    def __init__(self, config: SitrepConfig, client, **deps) -> None:
        """Store the config, sqlite client, and all wired components."""
        self.config = config
        self.client = client
        for key, value in deps.items():
            setattr(self, key, value)

    @property
    def retriever(self):
        """Convenience accessor for the hybrid retriever."""
        return self._retriever

    def stats(self) -> StatsDTO:
        """Compute aggregate system statistics."""
        from src.infrastructure.monitoring.metrics import get_metrics

        m = get_metrics()
        return StatsDTO(
            schemas=self._count("schemas"),
            facts=self._count("facts"),
            passages=self._count("passages"),
            episodes=self._count("episodes"),
            decisions=self._count("decisions"),
            feedback=self._count("feedback"),
            kv_caches=self._count("kv_cache"),
            fusion_weights=tuple(self._retriever.weights),
            token_reduction_ratio=m.token_reduction_ratio,
            tokens_saved=m.tokens_saved,
        )

    def _count(self, table: str) -> int:
        """Count rows in *table* (0 on error)."""
        try:
            row = self.client.fetchone(f"SELECT COUNT(*) AS c FROM {table}")
            return int(row["c"]) if row else 0
        except Exception:  # pragma: no cover
            return 0

    def close(self) -> None:
        """Release resources (sqlite connection)."""
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass


@log_execution
def build_application(config: Optional[SitrepConfig] = None) -> "Application":
    """Construct and wire the full application from *config* (or the global one)."""
    cfg = config or get_config(bootstrap=True)
    cfg.ensure_directories()

    # --- persistence ---
    from src.infrastructure.db.sqlite_client import SQLiteClient
    from src.adapters.repositories.sqlite_repo import (
        SQLiteDecisionRepo,
        SQLiteEpisodeRepo,
        SQLiteFactRepo,
        SQLiteFeedbackRepo,
        SQLitePassageRepo,
        SQLiteSchemaRepo,
        SQLiteSkillRepo,
    )
    from src.adapters.repositories.kv_cache_repo import SQLiteKVCacheRepository

    client = SQLiteClient.from_config()
    client.seed_fusion_weights(cfg.fusion_weights)
    schema_repo = SQLiteSchemaRepo(client)
    fact_repo = SQLiteFactRepo(client)
    passage_repo = SQLitePassageRepo(client)
    episode_repo = SQLiteEpisodeRepo(client)
    decision_repo = SQLiteDecisionRepo(client)
    skill_repo = SQLiteSkillRepo(client)
    feedback_repo = SQLiteFeedbackRepo(client)
    kv_repo = SQLiteKVCacheRepository(client)

    # --- gateways ---
    from src.adapters.services.embedding import EmbeddingService
    from src.infrastructure.embedding import get_embedding_gateway
    from src.infrastructure.llm import get_llm_gateway

    embedder = EmbeddingService(get_embedding_gateway(cfg))
    llm = get_llm_gateway(cfg)

    # --- services ---
    from src.adapters.services.classification import ClassificationService
    from src.adapters.services.compression import CompressionService
    from src.adapters.services.confidence import ConfidenceEstimator
    from src.adapters.services.conflict import (
        ConflictDetectionService,
        ConflictResolutionService,
    )
    from src.adapters.services.extraction import ExtractionService
    from src.adapters.services.quality import QualityEstimator
    from src.infrastructure.chunking import AdaptiveChunker
    from src.infrastructure.retrieval import (
        CrossEncoderReranker,
        HybridRetriever,
        TemporalRetriever,
    )

    chunker = AdaptiveChunker(cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_size, embedder=embedder)
    extraction = ExtractionService(llm_gateway=llm)
    detector = ConflictDetectionService()
    resolver = ConflictResolutionService(llm_gateway=llm)
    compression = CompressionService(embedder=embedder, llm=llm)
    classifier = ClassificationService()
    confidence = ConfidenceEstimator()
    quality = QualityEstimator(embedder=embedder)
    reranker = CrossEncoderReranker(cfg.reranker_model, embedder=embedder)

    # --- optional stores (lazy; degrade to None when extras are absent) ---
    graph_store = None
    try:
        from src.infrastructure.db.kuzu_client import KuzuClient

        graph_store = KuzuClient(cfg.graph_dir)
    except Exception as exc:  # pragma: no cover
        _logger.info("graph store disabled: %s", exc)
    vector_store = None
    try:
        import chromadb  # type: ignore  # noqa: F401
        from src.infrastructure.db.chroma_client import ChromaClient

        vector_store = ChromaClient(cfg.vectors_dir)
    except Exception as exc:  # pragma: no cover
        _logger.info("vector store disabled: %s", exc)

    from src.infrastructure.retrieval.ppr import PPREngine
    from src.infrastructure.retrieval.temporal_scorer import TemporalScorer

    ppr_engine = PPREngine() if getattr(cfg, "ppr_enabled", True) else None
    temporal_scorer = TemporalScorer(
        strategy=getattr(cfg, "temporal_strategy", "exponential"),
        half_life_days=getattr(cfg, "temporal_half_life_days", 180.0),
    )
    from src.infrastructure.retrieval.entity_graph import EntityGraphBuilder

    def _entity_graph_provider():
        try:
            return EntityGraphBuilder.build(fact_repo.all_valid())
        except Exception:  # pragma: no cover
            return {}

    retriever = HybridRetriever(
        passage_repo=passage_repo,
        fact_repo=fact_repo,
        embedder=embedder,
        vector_store=vector_store,
        graph_store=graph_store,
        reranker=reranker,
        sqlite=client,
        top_k=cfg.top_k,
        ppr_engine=ppr_engine,
        temporal_scorer=temporal_scorer,
        ppr_alpha=cfg.ppr_alpha,
        ppr_gamma=cfg.ppr_gamma,
        ppr_max_iter=cfg.ppr_max_iter,
        ppr_tol=cfg.ppr_tol,
        ppr_weight=cfg.ppr_weight,
        density_weight=cfg.density_weight,
        bridge_theta=cfg.bridge_theta,
        bridge_degree=cfg.bridge_degree,
        temporal_weight=cfg.temporal_weight,
        entity_graph_provider=_entity_graph_provider if getattr(cfg, "entity_graph_enabled", True) else None,
        track_access=getattr(cfg, "track_access", False),
    )
    temporal = TemporalRetriever(fact_repo=fact_repo, embedder=embedder)

    # --- RL ---
    from src.infrastructure.rl.compression_agent import PPOCompressionAgent
    from src.infrastructure.rl.compression_env import CompressionEnv
    from src.infrastructure.rl.reward_model import LLMRewardModel

    reward = LLMRewardModel(llm=llm, embedder=embedder)
    queries = _collect_training_queries(client, passage_repo)
    policy_path = cfg.policies_dir / "ppo_policy"
    env = CompressionEnv(
        embedder=embedder,
        retriever=retriever,
        compression=compression,
        llm=llm,
        reward_model=reward,
        queries=queries,
        top_k=cfg.top_k,
        compression_min=cfg.compression_min,
        compression_max=cfg.compression_max,
    )
    agent = PPOCompressionAgent(
        env, config=cfg, policy_path=str(policy_path) if policy_path.exists() else None
    )

    # --- cross-cutting + use cases ---
    from src.adapters.presenters import QueryPresenter
    from src.application.active_learning import ActiveLearningService
    from src.application.conflict import (
        ConflictDetectionAgent,
        ConflictResolutionAgent,
        ConflictUseCase,
        TemporalAgent,
    )
    from src.application.feedback import FeedbackUseCase
    from src.application.ingest import ExtractionAgent, IngestDocumentUseCase
    from src.application.lineage import LineageAnalysisUseCase
    from src.application.query import QueryKnowledgeUseCase
    from src.application.train import TrainAgentUseCase
    from src.application.versioning import VersionManagerUseCase
    from src.infrastructure.lineage import LineageTracker
    from src.infrastructure.versioning import VersionManager

    lineage = LineageTracker(decision_repo=decision_repo, graph_store=graph_store)
    version_manager = VersionManager(cfg.base_dir)
    active = ActiveLearningService(llm=llm, threshold=cfg.confidence_threshold)

    extraction_agent = ExtractionAgent(chunker, extraction, embedder)
    ingest_uc = IngestDocumentUseCase(
        extraction_agent, passage_repo, fact_repo, schema_repo, episode_repo,
        classifier, detector, resolver, lineage, cfg, graph_store=graph_store,
    )
    presenter = lambda dto: QueryPresenter.present_result(
        dto.answer, dto.results, dto.confidence, dto.compression_ratio,
        dto.full_tokens, dto.compressed_tokens, dto.query_id,
    )
    # --- Headroom: content-aware compression + cache alignment + CCR ---
    from src.adapters.repositories.ccr_repo import SQLiteCCRRepository
    from src.infrastructure.cache import CacheAligner
    from src.infrastructure.compression import (
        CodeCompressor,
        ContentRouter,
        Kompress,
        LogCompressor,
        SmartCrusher,
    )

    ccr_repo = SQLiteCCRRepository(client, default_ttl=3600)
    try:
        ccr_repo.purge_expired()
    except Exception:  # pragma: no cover
        pass
    content_router = ContentRouter(
        smart_crusher=SmartCrusher(),
        code_compressor=CodeCompressor(),
        log_compressor=LogCompressor(),
        kompress=Kompress(compression),
    )
    cache_aligner = CacheAligner()
    query_uc = QueryKnowledgeUseCase(
        retriever, compression, agent, llm, confidence, quality, active,
        lineage, client, cfg, presenter=presenter,
        content_router=content_router, cache_aligner=cache_aligner, ccr_repo=ccr_repo,
    )
    detection_agent = ConflictDetectionAgent(fact_repo, detector)
    resolution_agent = ConflictResolutionAgent(fact_repo, passage_repo, resolver)
    temporal_agent = TemporalAgent(temporal)
    conflict_uc = ConflictUseCase(detection_agent, resolution_agent)
    feedback_uc = FeedbackUseCase(feedback_repo, retriever, client, lineage, cfg)
    train_uc = TrainAgentUseCase(env, agent, reward, lineage, cfg)
    version_uc = VersionManagerUseCase(version_manager, lineage)
    lineage_uc = LineageAnalysisUseCase(lineage)

    # --- Phase B: memory hygiene (importance → consolidation → forgetting) ---
    from src.adapters.services.importance import ImportanceScorer
    from src.application.consolidation import ConsolidationUseCase
    from src.application.forgetting import ForgettingUseCase
    from src.domain.forgetting import ForgettingCriteria

    importance_scorer = ImportanceScorer(
        recency_w=cfg.importance_recency_w,
        frequency_w=cfg.importance_frequency_w,
        redundancy_w=cfg.importance_redundancy_w,
        source_w=cfg.importance_source_w,
    )
    forgetting_criteria = ForgettingCriteria(
        max_age_days=cfg.forgetting_max_age_days,
        inactive_period_days=cfg.forgetting_inactive_period_days,
        min_importance_score=cfg.forgetting_min_importance,
        importance_decay_rate=cfg.forgetting_decay_rate,
        min_access_frequency=cfg.forgetting_min_access_frequency,
        redundancy_threshold=cfg.forgetting_redundancy_threshold,
        min_redundancy_count=cfg.forgetting_min_redundancy_count,
        target_memory_size=cfg.forgetting_target_memory_size,
    )
    consolidation_uc = ConsolidationUseCase(
        passage_repo, importance_scorer, lineage,
        theta=cfg.consolidation_theta, max_pairs=cfg.consolidation_max_pairs,
    )
    forgetting_uc = ForgettingUseCase(passage_repo, importance_scorer, lineage, forgetting_criteria)

    # --- Phase C: write safety (event log + unit of work) ---
    from src.infrastructure.event_bus import get_event_bus
    from src.infrastructure.orchestrator import UnitOfWork
    from src.infrastructure.persistence import EventLog

    event_log = EventLog(cfg.logs_dir / "events.wal") if getattr(cfg, "event_log_enabled", True) else None
    if event_log is not None:
        event_log.subscribe(get_event_bus())

    def unit_of_work():
        return UnitOfWork(client, graph_store, vector_store)

    # --- Phase G: multimodal (images/audio/video) + causal ---
    from src.adapters.repositories.multimodal_repo import SQLiteMultimodalRepo
    from src.application.causal import CausalQueryUseCase
    from src.application.multimodal import MultimodalUseCase
    from src.domain.causal import CausalGraph
    from src.infrastructure.embedding.av import AVEmbedder
    from src.infrastructure.embedding.clip import CLIPEmbedder
    from src.infrastructure.reasoning.causal import CausalEngine

    mm_repo = SQLiteMultimodalRepo(client)
    clip_embedder = CLIPEmbedder()
    av_embedder = AVEmbedder()
    multimodal_uc = MultimodalUseCase(mm_repo, clip_embedder, av_embedder)
    causal_uc = CausalQueryUseCase(CausalEngine(CausalGraph()))

    # --- Phase D + E: query intelligence + quality loop (explanation / judge / cache) ---
    from src.adapters.services.explanation import ExplanationService
    from src.adapters.services.judge import LLMJudge
    from src.application.judge import JudgeUseCase
    from src.application.query_orchestrator import QueryOrchestrator
    from src.application.query_processing import (
        IntentClassifier,
        MultiHopReasoner,
        QueryDecomposer,
    )
    from src.infrastructure.cache.result_cache import (
        QueryFrequencyTracker,
        ResultCache,
    )

    explanation_service = ExplanationService(llm=llm)
    llm_judge = LLMJudge(llm=llm, embedder=embedder)
    judge_uc = JudgeUseCase(llm_judge, retriever=retriever)
    result_cache = ResultCache(ttl=cfg.result_cache_ttl)
    result_cache.subscribe(get_event_bus())
    frequency_tracker = QueryFrequencyTracker()
    query_orchestrator = QueryOrchestrator(
        query_uc,
        IntentClassifier(llm=llm),
        QueryDecomposer(llm=llm),
        MultiHopReasoner(fact_repo, retriever, llm, embedder),
        cache=result_cache,
        tracker=frequency_tracker,
        explanation_service=explanation_service,
    )

    # --- Phase F: anomaly detection + Allen temporal reasoning ---
    from src.adapters.services.anomaly import AnomalyDetector
    from src.application.anomaly import AnomalyUseCase
    from src.application.temporal_reasoning import TemporalReasoningUseCase

    anomaly_uc = AnomalyUseCase(passage_repo, AnomalyDetector())
    temporal_reasoning_uc = TemporalReasoningUseCase(fact_repo)

    return Application(
        cfg,
        client,
        _retriever=retriever,
        _llm=llm,
        _embedder=embedder,
        _agent=agent,
        _env=env,
        _reward=reward,
        _graph_store=graph_store,
        _vector_store=vector_store,
        kv_repo=kv_repo,
        ccr_repo=ccr_repo,
        content_router=content_router,
        cache_aligner=cache_aligner,
        passage_repo=passage_repo,
        fact_repo=fact_repo,
        schema_repo=schema_repo,
        episode_repo=episode_repo,
        feedback_repo=feedback_repo,
        decision_repo=decision_repo,
        skill_repo=skill_repo,
        version_manager=version_manager,
        lineage=lineage,
        ingest_uc=ingest_uc,
        query_uc=query_uc,
        conflict_uc=conflict_uc,
        feedback_uc=feedback_uc,
        train_uc=train_uc,
        version_uc=version_uc,
        lineage_uc=lineage_uc,
        importance_scorer=importance_scorer,
        consolidation_uc=consolidation_uc,
        forgetting_uc=forgetting_uc,
        forgetting_criteria=forgetting_criteria,
        event_log=event_log,
        unit_of_work=unit_of_work,
        multimodal_uc=multimodal_uc,
        causal_uc=causal_uc,
        query_orchestrator=query_orchestrator,
        explanation_service=explanation_service,
        judge_uc=judge_uc,
        result_cache=result_cache,
        frequency_tracker=frequency_tracker,
        llm_judge=llm_judge,
        anomaly_uc=anomaly_uc,
        temporal_reasoning_uc=temporal_reasoning_uc,
        mm_repo=mm_repo,
        clip_embedder=clip_embedder,
        av_embedder=av_embedder,
        temporal_agent=temporal_agent,
        detection_agent=detection_agent,
        resolution_agent=resolution_agent,
        extraction_agent=extraction_agent,
    )


def _collect_training_queries(client, passage_repo) -> list:
    """Build a query list for RL training from past queries or passages."""
    queries: list = []
    try:
        rows = client.fetchall(
            "SELECT DISTINCT query_text FROM retrieval_stats "
            "WHERE query_text IS NOT NULL ORDER BY created_at DESC LIMIT 50"
        )
        queries = [r["query_text"] for r in rows if r["query_text"]]
    except Exception:  # pragma: no cover
        queries = []
    if queries:
        return queries
    # Fall back to first-sentence pseudo-queries from stored passages.
    for pid in passage_repo.all_ids()[:50]:
        passage = passage_repo.get(pid)
        if passage:
            first = passage.text.split(".")[0][:80].strip()
            if first:
                queries.append(first)
    return queries or ["summarize the available context"]


__all__ = ["Application", "build_application"]
