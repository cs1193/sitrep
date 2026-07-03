"""SQLite-backed KV-cache repository (pickled transformer KV caches as BLOBs)."""
from __future__ import annotations

import logging
import pickle
import sqlite3
from typing import Any, Dict, List, Optional, Sequence

from src.domain.interfaces import KVCacheRepository
from src.infrastructure.db.sqlite_client import SQLiteClient
from src.utils.common import utc_now_iso

_logger = logging.getLogger("sitrep.repos.kv_cache")


class SQLiteKVCacheRepository(KVCacheRepository):
    """Stores precomputed KV caches keyed by passage id."""

    def __init__(self, client: SQLiteClient) -> None:
        self.client = client

    def has(self, passage_id: str) -> bool:
        """Return True if a cache exists for *passage_id*."""
        row = self.client.fetchone("SELECT 1 FROM kv_cache WHERE passage_id=?", (passage_id,))
        return row is not None

    def get(self, passage_id: str) -> Optional[Any]:
        """Return the deserialized cache for *passage_id* (or None)."""
        row = self.client.fetchone("SELECT cache FROM kv_cache WHERE passage_id=?", (passage_id,))
        if row is None or row["cache"] is None:
            return None
        try:
            return pickle.loads(bytes(row["cache"]))
        except Exception as exc:  # pragma: no cover
            _logger.warning("failed to unpickle kv cache for %s: %s", passage_id, exc)
            return None

    def store(
        self, passage_id: str, cache: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist *cache* for *passage_id* (pickled), recording *metadata*."""
        meta = metadata or {}
        blob = sqlite3.Binary(pickle.dumps(cache))
        with self.client.transaction():
            self.client.execute(
                "INSERT OR REPLACE INTO kv_cache "
                "(passage_id, cache, model, dim, layer_count, created_at, metadata) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    passage_id,
                    blob,
                    meta.get("model"),
                    meta.get("dim"),
                    meta.get("layer_count"),
                    utc_now_iso(),
                    self.client.dumps_json(meta),
                ),
            )
        _logger.debug("stored kv cache for %s", passage_id)

    def delete(self, passage_id: str) -> None:
        """Delete the cache for *passage_id*."""
        with self.client.transaction():
            self.client.execute("DELETE FROM kv_cache WHERE passage_id=?", (passage_id,))

    def missing(self, passage_ids: Sequence[str]) -> List[str]:
        """Return the subset of *passage_ids* lacking a cached entry."""
        if not passage_ids:
            return []
        placeholders = ",".join("?" * len(passages := list(passage_ids)))
        rows = self.client.fetchall(
            f"SELECT passage_id FROM kv_cache WHERE passage_id IN ({placeholders})", passages
        )
        present = {r["passage_id"] for r in rows}
        return [pid for pid in passages if pid not in present]

    def count(self) -> int:
        """Return the number of cached passages."""
        row = self.client.fetchone("SELECT COUNT(*) AS c FROM kv_cache")
        return int(row["c"]) if row else 0
