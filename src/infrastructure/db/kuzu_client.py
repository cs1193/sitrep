"""KuzuDB graph client (lazy).

Used for both the knowledge graph (``.sitrep/graph``) and the lineage graph
(```.sitrep/lineage``). ``kuzu`` is an optional dependency; this client degrades
to a no-op logger when it is absent so the rest of the system still runs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.common import utc_now_iso

_logger = logging.getLogger("sitrep.db.kuzu")

# Node/rel DDL. Kuzu's openCypher-flavoured DDL; created lazily on connect.
_NODE_DDL: Dict[str, str] = {
    "Schema": "CREATE NODE TABLE IF NOT EXISTS Schema(id STRING, name STRING, domain STRING, "
    "promoted BOOL, PRIMARY KEY(id))",
    "Fact": "CREATE NODE TABLE IF NOT EXISTS Fact(id STRING, subject STRING, predicate STRING, "
    "object STRING, valid_from STRING, valid_to STRING, status STRING, PRIMARY KEY(id))",
    "Passage": "CREATE NODE TABLE IF NOT EXISTS Passage(id STRING, document_id STRING, "
    "chunk_index INT64, PRIMARY KEY(id))",
    "Entity": "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING, kind STRING, name STRING, "
    "attrs STRING, PRIMARY KEY(id))",
    "Decision": "CREATE NODE TABLE IF NOT EXISTS Decision(id STRING, agent_id STRING, "
    "decision_type STRING, action STRING, timestamp STRING, PRIMARY KEY(id))",
}
_REL_DDL: List[str] = [
    "CREATE REL TABLE IF NOT EXISTS Contains(Fact, Passage)",
    "CREATE REL TABLE IF NOT EXISTS RelatesTo(Entity, Entity, kind STRING)",
    "CREATE REL TABLE IF NOT EXISTS Invalidates(Fact, Fact, when STRING, reason STRING)",
    "CREATE REL TABLE IF NOT EXISTS DerivedFrom(Fact, Fact)",
    "CREATE REL TABLE IF NOT EXISTS TriggeredBy(Decision, Decision)",
    "CREATE REL TABLE IF NOT EXISTS Used(Decision, Passage)",
]


class KuzuClient:
    """Lazy KuzuDB wrapper exposing Cypher execution + graph helpers.

    Parameters
    ----------
    db_dir:
        Directory holding the Kuzu database (e.g. ``.sitrep/graph``).
    """

    def __init__(self, db_dir: Union[str, Path]) -> None:
        """Store config; the database is opened lazily on first use."""
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._database: Any = None
        self._conn: Any = None
        self._initialized = False

    # ----------------------------------------------------------------- lifecycle
    def _ensure_open(self) -> None:
        """Open the database and connection, importing ``kuzu`` lazily."""
        if self._conn is not None:
            return
        try:
            import kuzu  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "kuzu is not installed. Install with: uv sync --extra graph"
            ) from exc
        self._database = kuzu.Database(str(self.db_dir))
        self._conn = kuzu.Connection(self._database)
        self._init_schema()
        self._initialized = True

    def _init_schema(self) -> None:
        """Create node/rel tables if absent (idempotent best-effort)."""
        for ddl in list(_NODE_DDL.values()) + _REL_DDL:
            try:
                self._conn.execute(ddl)
            except Exception as exc:  # kuzu raises if table already exists
                _logger.debug("kuzu DDL skipped (%s): %s", ddl.split()[3] if len(ddl.split()) > 3 else "?", exc)

    # ----------------------------------------------------------------- queries
    def execute(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher *query* and return rows as dicts."""
        self._ensure_open()
        try:
            result = self._conn.execute(cypher, params or {})
        except TypeError:
            # Older kuzu versions do not accept a params dict.
            result = self._conn.execute(cypher)
        rows: List[Dict[str, Any]] = []
        try:
            while result.hasNext():
                rows.append(dict(result.getNext()))
        except Exception:  # pragma: no cover - result sets vary across versions
            _logger.debug("kuzu result iteration ended")
        return rows

    # ----------------------------------------------------------------- helpers
    def add_entity(self, kind: str, properties: Dict[str, Any]) -> str:
        """Insert an ``Entity`` node of *kind* with stringified *properties*."""
        props = {"id": properties.get("id"), "kind": kind, "name": str(properties.get("name", "")),
                 "attrs": str(properties)}
        self.execute("MERGE (e:Entity {id: $id}) SET e.kind=$kind, e.name=$name, e.attrs=$attrs", props)
        return props["id"]

    def add_relation(
        self,
        src_kind: str,
        src_id: str,
        rel: str,
        dst_kind: str,
        dst_id: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a generic ``RelatesTo`` edge between two entities."""
        params = {"sid": src_id, "did": dst_id, "kind": str(rel)}
        self.execute(
            "MATCH (s:Entity {id:$sid}), (d:Entity {id:$did}) "
            "MERGE (s)-[r:RelatesTo]->(d) SET r.kind=$kind",
            params,
        )

    def neighbors(
        self, kind: str, entity_id: str, rel: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return neighbors of *entity_id* (optionally filtered by *rel* kind)."""
        rel_filter = f"WHERE r.kind = '{rel}'" if rel else ""
        cypher = (
            f"MATCH (e:Entity {{id:$id}})-[r:RelatesTo]-(n:Entity) {rel_filter} "
            f"RETURN n.id, n.kind, n.name LIMIT {int(limit)}"
        )
        return self.execute(cypher, {"id": entity_id})

    def density(self) -> float:
        """Return edges/float(nodes) as a crude graph density estimate."""
        nodes = self.execute("MATCH (n:Entity) RETURN count(n) AS c")
        edges = self.execute("MATCH ()-[r:RelatesTo]->() RETURN count(r) AS c")
        n = float(nodes[0].get("c", 0)) if nodes else 0.0
        e = float(edges[0].get("c", 0)) if edges else 0.0
        if n <= 1.0:
            return 0.0
        return min(1.0, e / (n * (n - 1.0) / 2.0))

    def close(self) -> None:
        """Close the connection (best-effort)."""
        self._conn = None
        self._database = None
        self._initialized = False

    def __enter__(self) -> "KuzuClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
