"""Port interfaces (abstract protocols) decoupling the domain from adapters.

These ``typing.Protocol`` classes define the contracts the application layer
depends on. Concrete adapters (SQLite, Kuzu, Chroma, Ollama/Transformers, ...)
implement them. Keeping ports here means the domain never imports a concrete
technology.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from src.domain.value_objects import RetrievalResult

# Entity types are imported only for annotations to avoid an import cycle.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.domain.schemas import (
        Agent,
        Decision,
        Episode,
        Fact,
        Passage,
        Schema,
        Skill,
    )


# =========================================================================== Repositories
@runtime_checkable
class SchemaRepository(Protocol):
    """Persistence port for :class:`Schema` aggregates."""

    def get(self, schema_id: str) -> Optional["Schema"]: ...

    def save(self, schema: "Schema") -> str: ...

    def list_all(self, promoted_only: bool = False) -> List["Schema"]: ...

    def find_by_name(self, name: str) -> Optional["Schema"]: ...

    def increment_usage(self, schema_id: str, by: int = 1) -> int: ...

    def promote_eligible(self, threshold: int) -> List[str]: ...


@runtime_checkable
class FactRepository(Protocol):
    """Persistence port for :class:`Fact` (and temporal queries)."""

    def get(self, fact_id: str) -> Optional["Fact"]: ...

    def save(self, fact: "Fact") -> str: ...

    def search(self, predicate: str, subject: Optional[str] = None) -> List["Fact"]: ...

    def find_conflicting(
        self, subject: str, predicate: str
    ) -> List["Fact"]: ...

    def invalidate(self, fact_id: str, when: Optional[datetime] = None, reason: str = "") -> None: ...

    def list_by_schema(self, schema_id: str) -> List["Fact"]: ...

    def point_in_time(self, moment: datetime) -> List["Fact"]: ...

    def count(self) -> int: ...


@runtime_checkable
class PassageRepository(Protocol):
    """Persistence port for :class:`Passage` (FTS + optional embedding)."""

    def get(self, passage_id: str) -> Optional["Passage"]: ...

    def save(self, passage: "Passage") -> str: ...

    def search_fts(self, query: str, limit: int = 10) -> List[Tuple["Passage", float]]: ...

    def list_by_document(self, document_id: str) -> List["Passage"]: ...

    def all_ids(self) -> List[str]: ...

    def count(self) -> int: ...


@runtime_checkable
class EpisodeRepository(Protocol):
    """Persistence port for :class:`Episode`."""

    def get(self, episode_id: str) -> Optional["Episode"]: ...

    def save(self, episode: "Episode") -> str: ...

    def list_all(self) -> List["Episode"]: ...


@runtime_checkable
class DecisionRepository(Protocol):
    """Persistence port for lineage :class:`Decision` records."""

    def save(self, decision: "Decision") -> str: ...

    def get(self, decision_id: str) -> Optional["Decision"]: ...

    def list_recent(self, limit: int = 50) -> List["Decision"]: ...

    def by_episode(self, episode_id: str) -> List["Decision"]: ...


@runtime_checkable
class SkillRepository(Protocol):
    """Persistence port for :class:`Skill`."""

    def get(self, skill_id: str) -> Optional["Skill"]: ...

    def save(self, skill: "Skill") -> str: ...

    def find_by_name(self, name: str) -> Optional["Skill"]: ...

    def list_all(self) -> List["Skill"]: ...


@runtime_checkable
class FeedbackRepository(Protocol):
    """Persistence port for user feedback ratings."""

    def save(
        self, query_id: str, polarity: str, rating: float, metadata: Optional[Dict[str, Any]] = None
    ) -> str: ...

    def list_recent(self, limit: int = 100) -> List[Dict[str, Any]]: ...


@runtime_checkable
class KVCacheRepository(Protocol):
    """Persistence port for precomputed transformer KV caches."""

    def has(self, passage_id: str) -> bool: ...

    def get(self, passage_id: str) -> Optional[Any]: ...

    def store(self, passage_id: str, cache: Any, metadata: Optional[Dict[str, Any]] = None) -> None: ...

    def delete(self, passage_id: str) -> None: ...

    def missing(self, passage_ids: Sequence[str]) -> List[str]: ...


# =========================================================================== Gateways / services
@runtime_checkable
class LLMGateway(Protocol):
    """Local LLM gateway (Ollama / Transformers / DEMO)."""

    name: str

    def is_available(self) -> bool: ...

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str: ...


@runtime_checkable
class EmbeddingGateway(Protocol):
    """Embedding gateway (sentence-transformers or hash fallback)."""

    dim: int

    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector store port (ChromaDB, lazy)."""

    def add(
        self,
        collection: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None: ...

    def query(
        self,
        collection: str,
        embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]: ...

    def count(self, collection: str) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph port (KuzuDB, lazy)."""

    def add_entity(self, kind: str, properties: Dict[str, Any]) -> str: ...

    def add_relation(
        self, src_kind: str, src_id: str, rel: str, dst_kind: str, dst_id: str, props: Optional[Dict[str, Any]] = None
    ) -> None: ...

    def neighbors(self, kind: str, entity_id: str, rel: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]: ...

    def density(self) -> float: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker port."""

    def rerank(self, query: str, documents: Sequence[str]) -> List[Tuple[int, float]]: ...


@runtime_checkable
class Retriever(Protocol):
    """Hybrid retrieval port returning fused :class:`RetrievalResult`."""

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]: ...

    def update_weights(self, weights: Tuple[float, float, float]) -> None: ...


@runtime_checkable
class CompressionPolicy(Protocol):
    """RL/heuristic policy selecting a compression ratio for a query."""

    def select_ratio(self, observation: Sequence[float]) -> float: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@runtime_checkable
class RewardModel(Protocol):
    """Reward model comparing compressed vs. full-context answers."""

    def score(
        self, query: str, compressed_answer: str, full_answer: str, context: Optional[str] = None
    ) -> float: ...
