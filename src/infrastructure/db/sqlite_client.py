"""SQLite metadata client (stdlib ``sqlite3`` + FTS5).

Owns the relational schema for schemas, facts, passages, episodes, agents,
decisions, skills, feedback, KV cache, lineage events, fusion weights and
retrieval statistics. FTS5 full-text indexes are created when the runtime
supports them; otherwise the client degrades to LIKE-based search.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from src.utils.common import utc_now_iso

_logger = logging.getLogger("sitrep.db.sqlite")

Params = Union[Sequence[Any], Dict[str, Any]]

# --------------------------------------------------------------------------- DDL
_SCHEMA_SQL: Tuple[str, ...] = (
    # ---- schemas ----
    """
    CREATE TABLE IF NOT EXISTS schemas (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT DEFAULT '',
        fields       TEXT DEFAULT '[]',
        domain       TEXT DEFAULT 'general',
        version      INTEGER DEFAULT 1,
        usage_count  INTEGER DEFAULT 0,
        is_promoted  INTEGER DEFAULT 0,
        created_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_schemas_name ON schemas(name)",
    "CREATE INDEX IF NOT EXISTS idx_schemas_promoted ON schemas(is_promoted)",
    # ---- facts ----
    """
    CREATE TABLE IF NOT EXISTS facts (
        id                  TEXT PRIMARY KEY,
        subject             TEXT NOT NULL,
        predicate           TEXT NOT NULL,
        object_value        TEXT NOT NULL,
        schema_id           TEXT,
        source_passage_ids  TEXT DEFAULT '[]',
        confidence          REAL DEFAULT 0.5,
        valid_from          TEXT,
        valid_to            TEXT,
        status              TEXT DEFAULT 'valid',
        episode_id          TEXT,
        attributes          TEXT DEFAULT '{}',
        created_at          TEXT NOT NULL,
        invalidated_at      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(subject, predicate)",
    "CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status)",
    "CREATE INDEX IF NOT EXISTS idx_facts_schema ON facts(schema_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_from, valid_to)",
    # ---- passages ----
    """
    CREATE TABLE IF NOT EXISTS passages (
        id           TEXT PRIMARY KEY,
        document_id  TEXT NOT NULL,
        text         TEXT NOT NULL,
        chunk_index  INTEGER DEFAULT 0,
        schema_ids   TEXT DEFAULT '[]',
        embedding    BLOB,
        tokens       INTEGER DEFAULT 0,
        metadata     TEXT DEFAULT '{}',
        created_at   TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(document_id)",
    # ---- episodes ----
    """
    CREATE TABLE IF NOT EXISTS episodes (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        description  TEXT DEFAULT '',
        fact_ids     TEXT DEFAULT '[]',
        start        TEXT,
        end          TEXT,
        attributes   TEXT DEFAULT '{}',
        created_at   TEXT NOT NULL
    )
    """,
    # ---- agents ----
    """
    CREATE TABLE IF NOT EXISTS agents (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        role         TEXT NOT NULL,
        config       TEXT DEFAULT '{}',
        active       INTEGER DEFAULT 1,
        created_at   TEXT NOT NULL
    )
    """,
    # ---- decisions (lineage mirror) ----
    """
    CREATE TABLE IF NOT EXISTS decisions (
        id             TEXT PRIMARY KEY,
        agent_id       TEXT,
        decision_type  TEXT,
        action         TEXT,
        inputs         TEXT DEFAULT '{}',
        outputs        TEXT DEFAULT '{}',
        rationale      TEXT DEFAULT '',
        episode_id     TEXT,
        metadata       TEXT DEFAULT '{}',
        timestamp      TEXT NOT NULL,
        lineage_ref    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_episode ON decisions(episode_id)",
    # ---- skills ----
    """
    CREATE TABLE IF NOT EXISTS skills (
        id             TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        description    TEXT DEFAULT '',
        prompt         TEXT DEFAULT '',
        version        INTEGER DEFAULT 1,
        usage_count    INTEGER DEFAULT 0,
        learned_params TEXT DEFAULT '{}',
        created_at     TEXT NOT NULL
    )
    """,
    # ---- feedback ----
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id           TEXT PRIMARY KEY,
        query_id     TEXT,
        polarity     TEXT,
        rating       REAL,
        metadata     TEXT DEFAULT '{}',
        created_at   TEXT NOT NULL
    )
    """,
    # ---- kv_cache ----
    """
    CREATE TABLE IF NOT EXISTS kv_cache (
        passage_id   TEXT PRIMARY KEY,
        cache        BLOB NOT NULL,
        model        TEXT,
        dim          INTEGER,
        layer_count  INTEGER,
        created_at   TEXT NOT NULL,
        metadata     TEXT DEFAULT '{}'
    )
    """,
    # ---- lineage_events (fine-grained operation log) ----
    """
    CREATE TABLE IF NOT EXISTS lineage_events (
        id           TEXT PRIMARY KEY,
        decision_id  TEXT,
        kind         TEXT,
        payload      TEXT DEFAULT '{}',
        created_at   TEXT NOT NULL
    )
    """,
    # ---- fusion_weights (single normalized row) ----
    """
    CREATE TABLE IF NOT EXISTS fusion_weights (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        bm25        REAL,
        vector      REAL,
        graph       REAL,
        updated_at  TEXT
    )
    """,
    # ---- retrieval_stats (for online fusion learning) ----
    """
    CREATE TABLE IF NOT EXISTS retrieval_stats (
        query_id      TEXT PRIMARY KEY,
        query_text    TEXT,
        bm25_scores   TEXT DEFAULT '[]',
        vector_scores TEXT DEFAULT '[]',
        graph_scores  TEXT DEFAULT '[]',
        feedback      REAL,
        created_at    TEXT NOT NULL
    )
    """,
)

_FTS_SQL: Tuple[str, ...] = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(passage_id, text, document_id)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id, subject, predicate, object_value)",
)


class SQLiteClient:
    """Thin, dependency-free wrapper over ``sqlite3`` for SITREP metadata."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Open (or create) the database at *db_path* and initialize the schema."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._fts5_available: Optional[bool] = None
        self.init_schema()
        _logger.debug("sqlite ready at %s (fts5=%s)", self.db_path, self.has_fts5)

    # ----------------------------------------------------------------- lifecycle
    @classmethod
    def from_config(cls) -> "SQLiteClient":
        """Construct a client using the configured ``metadata/sitrep.db`` path."""
        from src.utils.config import get_config

        cfg = get_config()
        cfg.ensure_directories()
        return cls(cfg.sqlite_db_path)

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying connection."""
        return self._conn

    @property
    def has_fts5(self) -> bool:
        """Return True if the SQLite runtime supports FTS5."""
        if self._fts5_available is None:
            try:
                cur = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='sqlitestat'"
                )
                cur.fetchall()
                self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)")
                self._conn.execute("DROP TABLE IF EXISTS __fts5_probe")
                self._fts5_available = True
            except sqlite3.OperationalError as exc:
                _logger.warning("FTS5 unavailable (%s); falling back to LIKE search", exc)
                self._fts5_available = False
        return self._fts5_available

    def init_schema(self) -> None:
        """Create all tables and indexes (idempotent)."""
        with self._conn:
            for stmt in _SCHEMA_SQL:
                self._conn.execute(stmt)
            if self.has_fts5:
                for stmt in _FTS_SQL:
                    try:
                        self._conn.execute(stmt)
                    except sqlite3.OperationalError as exc:  # pragma: no cover
                        _logger.warning("could not create FTS table: %s", exc)

    # ----------------------------------------------------------------- execution
    def execute(self, sql: str, params: Optional[Params] = None) -> sqlite3.Cursor:
        """Execute *sql* and return the cursor."""
        return self._conn.execute(sql, params or ())

    def executemany(self, sql: str, seq: Sequence[Params]) -> sqlite3.Cursor:
        """Execute *sql* against each parameter set in *seq*."""
        return self._conn.executemany(sql, seq)

    def fetchone(self, sql: str, params: Optional[Params] = None) -> Optional[sqlite3.Row]:
        """Execute *sql* and return a single row (or None)."""
        return self._conn.execute(sql, params or ()).fetchone()

    def fetchall(self, sql: str, params: Optional[Params] = None) -> List[sqlite3.Row]:
        """Execute *sql* and return all rows."""
        return self._conn.execute(sql, params or ()).fetchall()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager wrapping a transaction (commit on success, rollback on error)."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def dumps_json(value: Any) -> str:
        """Serialize *value* to a JSON string (safe for sqlite TEXT columns)."""
        return json.dumps(value, default=str, ensure_ascii=False)

    @staticmethod
    def loads_json(value: Optional[str], default: Any = None) -> Any:
        """Deserialize a JSON string from sqlite (returns *default* on falsy input)."""
        if not value:
            return default if default is not None else {}
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default if default is not None else {}

    def seed_fusion_weights(self, weights: Sequence[float]) -> None:
        """Ensure the single fusion-weights row exists with *weights* (bm25, vector, graph)."""
        assert len(weights) == 3, "expected 3 fusion weights"
        bm25, vec, graph = weights
        with self._conn:
            self._conn.execute(
                "INSERT INTO fusion_weights(id, bm25, vector, graph, updated_at) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET bm25=excluded.bm25, vector=excluded.vector, "
                "graph=excluded.graph, updated_at=excluded.updated_at",
                (bm25, vec, graph, utc_now_iso()),
            )

    def get_fusion_weights(self) -> Tuple[float, float, float]:
        """Return the persisted fusion weights, defaulting to a balanced triple."""
        row = self.fetchone("SELECT bm25, vector, graph FROM fusion_weights WHERE id=1")
        if row is None:
            return (1 / 3, 1 / 3, 1 / 3)
        return (float(row["bm25"]), float(row["vector"]), float(row["graph"]))

    # ----------------------------------------------------------------- dunder
    def __enter__(self) -> "SQLiteClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            _logger.debug("error closing sqlite connection", exc_info=True)
