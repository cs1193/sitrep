"""Reversible Compression (CCR) repository: store originals with a TTL.

Enables Headroom's reversible compression — the compressed context is sent to
the model, while the original is retained locally (default 1h TTL). The model
can request the uncompressed version via the ``retrieve`` tool using the
returned key.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.infrastructure.db.sqlite_client import SQLiteClient
from src.utils.common import hash_text, utc_now_iso

_logger = logging.getLogger("sitrep.repos.ccr")


class SQLiteCCRRepository:
    """SQLite-backed CCR store with per-entry TTL."""

    TABLE = "ccr_store"

    def __init__(self, client: SQLiteClient, default_ttl: int = 3600) -> None:
        """Wire the sqlite client and default TTL (seconds); create the table."""
        self.client = client
        self.default_ttl = int(default_ttl)
        self._init_table()

    # ----------------------------------------------------------------- schema
    def _init_table(self) -> None:
        """Create the CCR table and indexes (idempotent)."""
        with self.client.transaction():
            self.client.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    key           TEXT PRIMARY KEY,
                    content_hash  TEXT,
                    original      TEXT,
                    compressed    TEXT,
                    content_type  TEXT,
                    created_at    TEXT NOT NULL,
                    expires_at    TEXT NOT NULL,
                    metadata      TEXT
                )
                """
            )
            self.client.execute(f"CREATE INDEX IF NOT EXISTS idx_ccr_expires ON {self.TABLE}(expires_at)")

    # ----------------------------------------------------------------- ops
    def store(
        self,
        original: str,
        compressed: str,
        content_type: str = "text",
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an original/compressed pair; return the (content-derived) key.

        Same original content maps to the same key, refreshing its TTL.
        """
        key = "ccr_" + hash_text(original)[:24]
        now = datetime.now(timezone.utc)
        ttl_seconds = self.default_ttl if ttl is None else int(ttl)
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
        with self.client.transaction():
            self.client.execute(
                f"INSERT OR REPLACE INTO {self.TABLE} "
                "(key, content_hash, original, compressed, content_type, created_at, expires_at, metadata) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    key,
                    hash_text(original),
                    original,
                    compressed,
                    content_type,
                    now.isoformat(),
                    expires,
                    self.client.dumps_json(metadata or {}),
                ),
            )
        _logger.debug("CCR stored key=%s ttl=%ss content_type=%s", key, ttl_seconds, content_type)
        return key

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the stored entry if present and unexpired, else None (and purge)."""
        row = self.client.fetchone(
            f"SELECT * FROM {self.TABLE} WHERE key=?", (key,)
        )
        if row is None:
            return None
        if self._is_expired(row["expires_at"]):
            self.delete(key)
            _logger.debug("CCR key=%s expired", key)
            return None
        return {
            "key": row["key"],
            "original": row["original"],
            "compressed": row["compressed"],
            "content_type": row["content_type"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "metadata": self.client.loads_json(row["metadata"], {}),
        }

    def has(self, key: str) -> bool:
        """Return True if *key* exists and is unexpired."""
        return self.retrieve(key) is not None

    def delete(self, key: str) -> None:
        """Delete an entry by key."""
        with self.client.transaction():
            self.client.execute(f"DELETE FROM {self.TABLE} WHERE key=?", (key,))

    def purge_expired(self) -> int:
        """Delete all entries past their TTL; return the count removed."""
        now = utc_now_iso()
        with self.client.transaction():
            cur = self.client.execute(
                f"DELETE FROM {self.TABLE} WHERE expires_at < ?", (now,)
            )
        removed = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        if removed:
            _logger.info("CCR purged %d expired entries", removed)
        return removed

    def count(self) -> int:
        """Return the total number of stored entries (including expired)."""
        row = self.client.fetchone(f"SELECT COUNT(*) AS c FROM {self.TABLE}")
        return int(row["c"]) if row else 0

    @staticmethod
    def _is_expired(expires_at: str) -> bool:
        """Return True if *expires_at* (ISO) is in the past."""
        try:
            return datetime.fromisoformat(expires_at) < datetime.now(timezone.utc)
        except (TypeError, ValueError):
            return False
