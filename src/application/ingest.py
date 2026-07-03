"""Ingestion use case + the ExtractionAgent role.

The :class:`ExtractionAgent` chunks a document, embeds the chunks, and extracts
facts/schemas. :class:`IngestDocumentUseCase` persists everything, merges
schemas, runs conflict detection/resolution, groups facts into an episode, and
records lineage.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.adapters.services.classification import ClassificationService
from src.adapters.services.conflict import (
    Conflict,
    ConflictDetectionService,
    ConflictResolutionService,
)
from src.adapters.services.extraction import ExtractionResult, ExtractionService
from src.domain.interfaces import (
    EmbeddingGateway,
    EpisodeRepository,
    FactRepository,
    PassageRepository,
    SchemaRepository,
)
from src.domain.schemas import Episode, Fact, Passage, Schema
from src.application.dto import IngestResultDTO
from src.application.events import document_ingested
from src.infrastructure.chunking import AdaptiveChunker
from src.infrastructure.lineage import LineageTracker
from src.domain.schemas import Decision
from src.utils.common import generate_id
from src.utils.constants import DEC_CONFLICT, DEC_INGEST
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.usecase.ingest")


class ExtractionAgent:
    """Multi-agent role: chunk → embed → extract facts and an inferred schema."""

    def __init__(
        self,
        chunker: AdaptiveChunker,
        extraction_service: ExtractionService,
        embedder: EmbeddingGateway,
    ) -> None:
        """Wire the chunker, extraction service, and embedder."""
        self.chunker = chunker
        self.extraction_service = extraction_service
        self.embedder = embedder

    def run(
        self, text: str, document_id: str, domain: str = "general"
    ) -> Tuple[List[Passage], List[Fact], Optional[Schema], str]:
        """Process *text* into passages, facts, and an inferred schema."""
        chunks = self.chunker.chunk(text)
        passages: List[Passage] = []
        facts: List[Fact] = []
        predicates: List[str] = []
        method = "regex"
        for i, chunk in enumerate(chunks):
            passage = Passage(
                text=chunk,
                document_id=document_id,
                chunk_index=i,
                embedding=self.embedder.embed(chunk),
            )
            passages.append(passage)
            result: ExtractionResult = self.extraction_service.extract(
                chunk, source_passage_id=passage.id
            )
            if result.method == "llm":
                method = "llm"
            for fact in result.facts:
                fact.source_passage_ids = [passage.id]
                facts.append(fact)
                if fact.predicate not in predicates:
                    predicates.append(fact.predicate)
        schema = self._build_schema(domain, predicates) if facts else None
        _logger.info(
            "ExtractionAgent: %d passages, %d facts (domain=%s)", len(passages), len(facts), domain
        )
        return passages, facts, schema, method

    @staticmethod
    def _build_schema(domain: str, predicates: List[str]) -> Schema:
        """Construct a schema with one field per distinct predicate."""
        fields = [{"name": "subject", "type": "entity"}] + [
            {"name": p, "type": "text"} for p in predicates
        ]
        return Schema(
            name=f"{domain}_facts",
            description=f"Inferred schema for {domain} facts",
            fields=fields,
            domain=domain,
        )


class IngestDocumentUseCase:
    """Orchestrates document ingestion end-to-end."""

    def __init__(
        self,
        extraction_agent: ExtractionAgent,
        passage_repo: PassageRepository,
        fact_repo: FactRepository,
        schema_repo: SchemaRepository,
        episode_repo: EpisodeRepository,
        classifier: ClassificationService,
        conflict_detector: ConflictDetectionService,
        conflict_resolver: ConflictResolutionService,
        lineage_tracker: LineageTracker,
        config=None,
        graph_store=None,
    ) -> None:
        """Wire all dependencies required for ingestion."""
        self.extraction_agent = extraction_agent
        self.passage_repo = passage_repo
        self.fact_repo = fact_repo
        self.schema_repo = schema_repo
        self.episode_repo = episode_repo
        self.classifier = classifier
        self.conflict_detector = conflict_detector
        self.conflict_resolver = conflict_resolver
        self.lineage_tracker = lineage_tracker
        self.config = config
        self.graph_store = graph_store

    @log_execution
    def execute(self, text: str, document_id: Optional[str] = None) -> IngestResultDTO:
        """Ingest *text*, returning an :class:`IngestResultDTO` summary."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        doc_id = document_id or generate_id("doc")
        domain, _conf = self.classifier.classify(text)

        passages, facts, schema, method = self.extraction_agent.run(text, doc_id, domain)
        for passage in passages:
            self.passage_repo.save(passage)

        schema_count = 0
        schema_id: Optional[str] = None
        if schema is not None:
            existing = self.schema_repo.find_by_name(schema.name)
            if existing is not None:
                schema_id = existing.id
                self.schema_repo.increment_usage(existing.id, len(facts))
                threshold = getattr(self.config, "schema_promotion_threshold", 5)
                existing.usage_count += len(facts)
                existing.maybe_promote(threshold)
                self.schema_repo.save(existing)
            else:
                schema.usage_count = len(facts)
                self.schema_repo.save(schema)
                schema_id = schema.id
            schema_count = 1

        for fact in facts:
            fact.schema_id = schema_id
            self.fact_repo.save(fact)

        conflicts = self._resolve_conflicts()

        episode = Episode(
            name=f"ingest:{doc_id}",
            description=f"{len(facts)} facts from {len(passages)} passages",
            fact_ids=[f.id for f in facts],
        )
        self.episode_repo.save(episode)

        self.lineage_tracker.record(
            Decision(
                agent_id="extraction",
                decision_type=DEC_INGEST,
                action="ingest_document",
                inputs={"document_id": doc_id, "chars": len(text)},
                outputs={"passages": len(passages), "facts": len(facts), "conflicts": len(conflicts)},
                rationale=f"domain={domain}; schema={schema.name if schema else None}",
                episode_id=episode.id,
            )
        )
        document_ingested(doc_id, len(facts), len(passages), domain).publish()
        return IngestResultDTO(
            document_id=doc_id,
            passages=len(passages),
            facts=len(facts),
            schemas=schema_count,
            conflicts_detected=len(conflicts),
            conflicts_resolved=len(conflicts),
            domain=domain,
            method=method,
            episode_id=episode.id,
        )

    def _resolve_conflicts(self) -> List[Conflict]:
        """Detect conflicts among all valid facts and invalidate losers."""
        conflicts = self.conflict_detector.detect(self.fact_repo.all_valid())
        for conflict in conflicts:
            resolution = self.conflict_resolver.resolve(conflict)
            kept_id = resolution.kept_fact_ids[0] if resolution.kept_fact_ids else None
            for fid in resolution.invalidated_fact_ids:
                self.fact_repo.invalidate(fid, reason=conflict.description)
                self._record_provenance(loser_id=fid, kept_id=kept_id, conflict=conflict)
        return conflicts

    def _record_provenance(
        self, loser_id: str, kept_id: Optional[str], conflict: Conflict
    ) -> None:
        """Record SUPERSEDES/INVALIDATED_BY provenance.

        Writes ``superseded_by``/``supersedes`` on the facts, a SUPERSEDES
        decision in lineage, and (when a graph store is wired) a Kuzu edge.
        """
        if not kept_id:
            return
        try:
            loser = self.fact_repo.get(loser_id)
            if loser is not None:
                loser.attributes["superseded_by"] = kept_id
                loser.attributes["superseded_reason"] = conflict.description
                self.fact_repo.save(loser)
            kept = self.fact_repo.get(kept_id)
            if kept is not None:
                supersedes = list(kept.attributes.get("supersedes", []))
                if loser_id not in supersedes:
                    supersedes.append(loser_id)
                kept.attributes["supersedes"] = supersedes
                self.fact_repo.save(kept)
        except Exception as exc:  # pragma: no cover
            _logger.warning("provenance attribute write failed: %s", exc)
        self.lineage_tracker.record(
            Decision(
                agent_id="conflict",
                decision_type=DEC_CONFLICT,
                action="supersede",
                inputs={"conflict_type": conflict.conflict_type.value},
                outputs={"kept": kept_id, "invalidated": loser_id},
                rationale=conflict.description,
            )
        )
        if self.graph_store is not None:
            try:
                self.graph_store.add_entity("Fact", {"id": kept_id})
                self.graph_store.add_entity("Fact", {"id": loser_id})
                self.graph_store.add_relation(
                    "Fact", kept_id, "SUPERSEDES", "Fact", loser_id,
                    {"reason": conflict.description},
                )
            except Exception as exc:  # pragma: no cover
                _logger.debug("provenance graph edge skipped: %s", exc)
