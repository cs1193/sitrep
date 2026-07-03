"""DuckDB client (lazy) for Parquet/JSONL document archives.

Optional dependency (extra ``[duckdb]``). Used to materialize and query the
``.sitrep/documents`` archive (raw / chunks / archives).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_logger = logging.getLogger("sitrep.db.duckdb")


class DuckDBClient:
    """Lazy DuckDB wrapper for columnar document archives."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Configure the on-disk database path (``None`` → in-memory)."""
        self.db_path: Optional[Path] = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Any = None

    def _ensure_open(self) -> None:
        """Open the DuckDB connection lazily."""
        if self._conn is not None:
            return
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "duckdb is not installed. Install with: uv sync --extra duckdb"
            ) from exc
        self._conn = duckdb.connect(str(self.db_path) if self.db_path else ":memory:")
        _logger.info("duckdb opened (%s)", self.db_path or "memory")

    # ----------------------------------------------------------------- operations
    def write_passages_parquet(self, rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> Path:
        """Write *rows* to a Parquet file at *path* via DuckDB."""
        self._ensure_open()
        import json as _json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Serialize nested fields so Parquet stays flat.
        flat = [
            {k: (_json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
            for row in rows
        ]
        self._conn.register("_sitrep_rows", flat)
        self._conn.execute(f"COPY (SELECT * FROM _sitrep_rows) TO '{target}' (FORMAT PARQUET)")
        _logger.info("wrote %d rows → %s", len(flat), target)
        return target

    def write_jsonl(self, rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> Path:
        """Write *rows* as JSONL (no DuckDB required)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        return target

    def query_parquet(self, path: Union[str, Path], sql_select: str = "SELECT *") -> List[Dict[str, Any]]:
        """Run ``SELECT`` over a Parquet file and return rows as dicts."""
        self._ensure_open()
        cur = self._conn.execute(f"{sql_select} FROM read_parquet('{Path(path)}')")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[List[Any]]:
        """Execute arbitrary SQL and return rows."""
        self._ensure_open()
        cur = self._conn.execute(sql, list(params) if params else [])
        return [list(r) for r in cur.fetchall()]

    def close(self) -> None:
        """Close the connection."""
        self._conn = None

    def __enter__(self) -> "DuckDBClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
