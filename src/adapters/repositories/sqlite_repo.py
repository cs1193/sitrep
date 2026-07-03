"""SQLite-backed repositories implementing the domain repository ports.

Each class wraps a :class:`~src.infrastructure.db.sqlite_client.SQLiteClient`
and maps between rows and domain entities. JSON columns are serialized with the
client's ``dumps_json``/``loads_json`` helpers; embeddings are stored as pickled
BLOBs.
"""
from __future__ import annotations

import logging
import pickle
import re
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from src.domain.schemas import Decision, Episode, Fact, Passage, Schema, Skill
from src.domain.value_objects import FactStatus
from src.infrastructure.db.sqlite_client import SQLiteClient
from src.utils.common import generate_id, utc_now_iso

_logger = logging.getLogger("sitrep.repos.sqlite")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(text: str, max_terms: int = 20) -> Optional[str]:
    """Build a safe FTS5 MATCH expression (quoted tokens joined by ``OR``)."""
    toks = _TOKEN_RE.findall((text or "").lower())
    if not toks:
        return None
    unique = list(dict.fromkeys(toks))[:max_terms]
    return " OR ".join(f'"{t}"' for t in unique)


# =========================================================================== Schema
class SQLiteSchemaRepo:
    """Schema aggregate persistence."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    @staticmethod
    def _row_to_schema(row: sqlite3.Row) -> Schema:
        return Schema.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "fields": SQLiteClient.loads_json(row["fields"], []),
                "domain": row["domain"],
                "version": row["version"],
                "usage_count": row["usage_count"],
                "is_promoted": bool(row["is_promoted"]),
                "created_at": row["created_at"],
            }
        )

    def get(self, schema_id: str) -> Optional[Schema]:
        row = self.client.fetchone("SELECT * FROM schemas WHERE id=?", (schema_id,))
        return self._row_to_schema(row) if row else None

    def save(self, schema: Schema) -> str:
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO schemas "
                "(id, name, description, fields, domain, version, usage_count, is_promoted, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    schema.id,
                    schema.name,
                    schema.description,
                    self.client.dumps_json(schema.fields),
                    schema.domain,
                    schema.version,
                    schema.usage_count,
                    int(schema.is_promoted),
                    schema.created_at,
                ),
            )
        return schema.id

    def list_all(self, promoted_only: bool = False) -> List[Schema]:
        sql = "SELECT * FROM schemas"
        if promoted_only:
            sql += " WHERE is_promoted=1"
        sql += " ORDER BY usage_count DESC"
        return [self._row_to_schema(r) for r in self.client.fetchall(sql)]

    def find_by_name(self, name: str) -> Optional[Schema]:
        row = self.client.fetchone("SELECT * FROM schemas WHERE name=? LIMIT 1", (name,))
        return self._row_to_schema(row) if row else None

    def increment_usage(self, schema_id: str, by: int = 1) -> int:
        with self.client.transaction():
            self.client.execute(
                "UPDATE schemas SET usage_count=usage_count+? WHERE id=?", (by, schema_id)
            )
        row = self.client.fetchone("SELECT usage_count FROM schemas WHERE id=?", (schema_id,))
        return int(row["usage_count"]) if row else 0

    def promote_eligible(self, threshold: int) -> List[str]:
        rows = self.client.fetchall(
            "SELECT id FROM schemas WHERE usage_count>=? AND is_promoted=0", (threshold,)
        )
        ids = [r["id"] for r in rows]
        if ids:
            with self.client.transaction():
                placeholders = ",".join("?" * len(ids))
                self.client.execute(
                    f"UPDATE schemas SET is_promoted=1 WHERE id IN ({placeholders})", ids
                )
        return ids


# =========================================================================== Fact
class SQLiteFactRepo:
    """Fact persistence with bi-temporal queries and FTS5 search."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact.from_dict(
            {
                "id": row["id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object_value": row["object_value"],
                "schema_id": row["schema_id"],
                "source_passage_ids": SQLiteClient.loads_json(row["source_passage_ids"], []),
                "confidence": row["confidence"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "status": row["status"],
                "episode_id": row["episode_id"],
                "attributes": SQLiteClient.loads_json(row["attributes"], {}),
                "created_at": row["created_at"],
                "invalidated_at": row["invalidated_at"],
            }
        )

    def get(self, fact_id: str) -> Optional[Fact]:
        row = self.client.fetchone("SELECT * FROM facts WHERE id=?", (fact_id,))
        return self._row_to_fact(row) if row else None

    def save(self, fact: Fact) -> str:
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO facts "
                "(id, subject, predicate, object_value, schema_id, source_passage_ids, confidence, "
                "valid_from, valid_to, status, episode_id, attributes, created_at, invalidated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fact.id,
                    fact.subject,
                    fact.predicate,
                    fact.object_value,
                    fact.schema_id,
                    self.client.dumps_json(fact.source_passage_ids),
                    fact.confidence,
                    fact.valid_from,
                    fact.valid_to,
                    fact.status.value,
                    fact.episode_id,
                    self.client.dumps_json(fact.attributes),
                    fact.created_at,
                    fact.invalidated_at,
                ),
            )
            if self.client.has_fts5:
                self.client.execute(
                    "INSERT INTO facts_fts(fact_id, subject, predicate, object_value) VALUES (?,?,?,?)",
                    (fact.id, fact.subject, fact.predicate, fact.object_value),
                )
        return fact.id

    def search(self, predicate: str, subject: Optional[str] = None) -> List[Fact]:
        """Full-text search across fact fields (optionally constrained by *subject*)."""
        fts = _fts_query(predicate)
        facts: List[Fact] = []
        if self.client.has_fts5 and fts:
            try:
                rows = self.client.fetchall(
                    "SELECT fact_id, bm25(facts_fts) AS rank FROM facts_fts "
                    "WHERE facts_fts MATCH ? ORDER BY rank LIMIT 50",
                    (fts,),
                )
                for r in rows:
                    f = self.get(r["fact_id"])
                    if f and (subject is None or f.subject.lower() == subject.lower()):
                        facts.append(f)
            except sqlite3.OperationalError:
                facts = []
        if not facts:
            like = f"%{predicate}%"
            rows = self.client.fetchall(
                "SELECT * FROM facts WHERE (subject LIKE ? OR predicate LIKE ? OR object_value LIKE ?) "
                "AND status='valid' LIMIT 50",
                (like, like, like),
            )
            for r in rows:
                f = self._row_to_fact(r)
                if subject is None or f.subject.lower() == subject.lower():
                    facts.append(f)
        return facts

    def find_conflicting(self, subject: str, predicate: str) -> List[Fact]:
        rows = self.client.fetchall(
            "SELECT * FROM facts WHERE subject=? AND predicate=? AND status='valid'",
            (subject, predicate),
        )
        return [self._row_to_fact(r) for r in rows]

    def invalidate(
        self, fact_id: str, when: Optional[datetime] = None, reason: str = ""
    ) -> None:
        moment = (when or datetime.utcnow()).isoformat()
        with self.client.transaction():
            self.client.execute(
                "UPDATE facts SET status=?, valid_to=COALESCE(valid_to,?), invalidated_at=? WHERE id=?",
                (FactStatus.INVALIDATED.value, moment, moment, fact_id),
            )

    def list_by_schema(self, schema_id: str) -> List[Fact]:
        rows = self.client.fetchall("SELECT * FROM facts WHERE schema_id=?", (schema_id,))
        return [self._row_to_fact(r) for r in rows]

    def all_valid(self) -> List[Fact]:
        rows = self.client.fetchall("SELECT * FROM facts WHERE status='valid'")
        return [self._row_to_fact(r) for r in rows]

    def point_in_time(self, moment: datetime) -> List[Fact]:
        """Return facts valid (and recorded) at *moment*."""
        iso = moment.isoformat()
        rows = self.client.fetchall(
            "SELECT * FROM facts WHERE status='valid' AND created_at<=? "
            "AND valid_from<=? AND (valid_to IS NULL OR valid_to>?)",
            (iso, iso, iso),
        )
        return [self._row_to_fact(r) for r in rows]

    def count(self) -> int:
        row = self.client.fetchone("SELECT COUNT(*) AS c FROM facts")
        return int(row["c"]) if row else 0


# =========================================================================== Passage
class SQLitePassageRepo:
    """Passage persistence with FTS5 + pickled embedding BLOBs."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    @staticmethod
    def _row_to_passage(row: sqlite3.Row, include_embedding: bool = False) -> Passage:
        embedding = None
        if include_embedding and row["embedding"] is not None:
            try:
                embedding = pickle.loads(bytes(row["embedding"]))
            except Exception:  # pragma: no cover
                embedding = None
        return Passage.from_dict(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "text": row["text"],
                "chunk_index": row["chunk_index"],
                "schema_ids": SQLiteClient.loads_json(row["schema_ids"], []),
                "embedding": embedding,
                "tokens": row["tokens"],
                "metadata": SQLiteClient.loads_json(row["metadata"], {}),
                "created_at": row["created_at"],
            }
        )

    def get(self, passage_id: str) -> Optional[Passage]:
        row = self.client.fetchone("SELECT * FROM passages WHERE id=?", (passage_id,))
        return self._row_to_passage(row) if row else None

    def save(self, passage: Passage) -> str:
        emb_blob = sqlite3.Binary(pickle.dumps(list(passage.embedding))) if passage.embedding else None
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO passages "
                "(id, document_id, text, chunk_index, schema_ids, embedding, tokens, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    passage.id,
                    passage.document_id,
                    passage.text,
                    passage.chunk_index,
                    self.client.dumps_json(passage.schema_ids),
                    emb_blob,
                    passage.tokens,
                    self.client.dumps_json(passage.metadata),
                    passage.created_at,
                ),
            )
            if self.client.has_fts5:
                self.client.execute(
                    "INSERT INTO passages_fts(passage_id, text, document_id) VALUES (?,?,?)",
                    (passage.id, passage.text, passage.document_id),
                )
        return passage.id

    def search_fts(self, query: str, limit: int = 10) -> List[Tuple[Passage, float]]:
        """FTS5/BM25 passage search with a LIKE fallback."""
        fts = _fts_query(query)
        results: List[Tuple[Passage, float]] = []
        if self.client.has_fts5 and fts:
            try:
                rows = self.client.fetchall(
                    "SELECT passage_id, bm25(passages_fts) AS rank FROM passages_fts "
                    "WHERE passages_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts, limit),
                )
                for r in rows:
                    p = self.get(r["passage_id"])
                    if p:
                        results.append((p, max(0.0, -float(r["rank"]))))
            except sqlite3.OperationalError:
                results = []
        if not results:
            like = f"%{query[:120]}%"
            rows = self.client.fetchall(
                "SELECT * FROM passages WHERE text LIKE ? LIMIT ?", (like, limit)
            )
            for r in rows:
                results.append((self._row_to_passage(r), 1.0))
        return results

    def list_by_document(self, document_id: str) -> List[Passage]:
        rows = self.client.fetchall(
            "SELECT * FROM passages WHERE document_id=? ORDER BY chunk_index", (document_id,)
        )
        return [self._row_to_passage(r) for r in rows]

    def all_ids(self) -> List[str]:
        rows = self.client.fetchall("SELECT id FROM passages")
        return [r["id"] for r in rows]

    def iter_with_embeddings(self) -> Iterator[Passage]:
        """Yield passages that have a stored embedding (vector fallback path)."""
        rows = self.client.fetchall("SELECT * FROM passages WHERE embedding IS NOT NULL")
        for r in rows:
            yield self._row_to_passage(r, include_embedding=True)

    def count(self) -> int:
        row = self.client.fetchone("SELECT COUNT(*) AS c FROM passages")
        return int(row["c"]) if row else 0

    def iter_all(self) -> Iterator[Passage]:
        """Yield every passage (metadata only, no embedding) — for memory hygiene."""
        rows = self.client.fetchall("SELECT * FROM passages")
        for r in rows:
            yield self._row_to_passage(r, include_embedding=False)

    def find_near_duplicates(
        self,
        embedding: Sequence[float],
        theta: float = 0.85,
        limit: int = 5,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[Passage, float]]:
        """Return ``(passage, cosine_similarity)`` pairs with similarity ≥ *theta*."""
        from src.utils.common import cosine_similarity

        hits: List[Tuple[Passage, float]] = []
        for passage in self.iter_with_embeddings():
            if passage.embedding is None or (exclude_id and passage.id == exclude_id):
                continue
            sim = cosine_similarity(embedding, passage.embedding)
            if sim >= theta:
                hits.append((passage, float(sim)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:limit]


# =========================================================================== Episode
class SQLiteEpisodeRepo:
    """Episode persistence."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    def get(self, episode_id: str) -> Optional[Episode]:
        row = self.client.fetchone("SELECT * FROM episodes WHERE id=?", (episode_id,))
        if not row:
            return None
        return Episode(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            fact_ids=SQLiteClient.loads_json(row["fact_ids"], []),
            start=row["start"],
            end=row["end"],
            attributes=SQLiteClient.loads_json(row["attributes"], {}),
            created_at=row["created_at"],
        )

    def save(self, episode: Episode) -> str:
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, name, description, fact_ids, start, end, attributes, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    episode.id,
                    episode.name,
                    episode.description,
                    self.client.dumps_json(episode.fact_ids),
                    episode.start,
                    episode.end,
                    self.client.dumps_json(episode.attributes),
                    episode.created_at,
                ),
            )
        return episode.id

    def list_all(self) -> List[Episode]:
        return [self.get(r["id"]) for r in self.client.fetchall("SELECT id FROM episodes")]


# =========================================================================== Decision (lineage)
class SQLiteDecisionRepo:
    """Decision persistence (lineage mirror)."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    def save(self, decision: Decision) -> str:
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO decisions "
                "(id, agent_id, decision_type, action, inputs, outputs, rationale, episode_id, "
                "metadata, timestamp, lineage_ref) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.id,
                    decision.agent_id,
                    decision.decision_type,
                    decision.action,
                    self.client.dumps_json(decision.inputs),
                    self.client.dumps_json(decision.outputs),
                    decision.rationale,
                    decision.episode_id,
                    self.client.dumps_json(decision.metadata),
                    decision.timestamp,
                    decision.lineage_ref,
                ),
            )
        return decision.id

    def get(self, decision_id: str) -> Optional[Decision]:
        row = self.client.fetchone("SELECT * FROM decisions WHERE id=?", (decision_id,))
        if not row:
            return None
        return Decision(
            id=row["id"],
            agent_id=row["agent_id"],
            decision_type=row["decision_type"],
            action=row["action"],
            inputs=SQLiteClient.loads_json(row["inputs"], {}),
            outputs=SQLiteClient.loads_json(row["outputs"], {}),
            rationale=row["rationale"],
            episode_id=row["episode_id"],
            metadata=SQLiteClient.loads_json(row["metadata"], {}),
            timestamp=row["timestamp"],
            lineage_ref=row["lineage_ref"],
        )

    def list_recent(self, limit: int = 50) -> List[Decision]:
        rows = self.client.fetchall(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [self.get(r["id"]) for r in rows]  # type: ignore[arg-type]

    def by_episode(self, episode_id: str) -> List[Decision]:
        rows = self.client.fetchall("SELECT * FROM decisions WHERE episode_id=?", (episode_id,))
        return [self.get(r["id"]) for r in rows]  # type: ignore[arg-type]


# =========================================================================== Skill
class SQLiteSkillRepo:
    """Skill persistence."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    def get(self, skill_id: str) -> Optional[Skill]:
        row = self.client.fetchone("SELECT * FROM skills WHERE id=?", (skill_id,))
        if not row:
            return None
        return Skill(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            prompt=row["prompt"],
            version=row["version"],
            usage_count=row["usage_count"],
            learned_params=SQLiteClient.loads_json(row["learned_params"], {}),
            created_at=row["created_at"],
        )

    def save(self, skill: Skill) -> str:
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO skills "
                "(id, name, description, prompt, version, usage_count, learned_params, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    skill.id,
                    skill.name,
                    skill.description,
                    skill.prompt,
                    skill.version,
                    skill.usage_count,
                    self.client.dumps_json(skill.learned_params),
                    skill.created_at,
                ),
            )
        return skill.id

    def find_by_name(self, name: str) -> Optional[Skill]:
        row = self.client.fetchone("SELECT * FROM skills WHERE name=? LIMIT 1", (name,))
        return self.get(row["id"]) if row else None

    def list_all(self) -> List[Skill]:
        return [self.get(r["id"]) for r in self.client.fetchall("SELECT id FROM skills")]


# =========================================================================== Feedback
class SQLiteFeedbackRepo:
    """User-feedback persistence."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    def save(
        self,
        query_id: str,
        polarity: str,
        rating: float,
        metadata: Optional[dict] = None,
    ) -> str:
        fid = generate_id("feedback")
        with self.client.transaction():
            self.client.execute(
                "INSERT INTO feedback(id, query_id, polarity, rating, metadata, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    fid,
                    query_id,
                    polarity,
                    float(rating),
                    self.client.dumps_json(metadata or {}),
                    utc_now_iso(),
                ),
            )
        return fid

    def list_recent(self, limit: int = 100) -> List[dict]:
        rows = self.client.fetchall(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [
            {
                "id": r["id"],
                "query_id": r["query_id"],
                "polarity": r["polarity"],
                "rating": r["rating"],
                "metadata": SQLiteClient.loads_json(r["metadata"], {}),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
