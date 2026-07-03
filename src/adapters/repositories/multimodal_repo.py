"""SQLite repository for multimodal assets (images, cross-modal links, media)."""
from __future__ import annotations

import logging
import pickle
import sqlite3
from typing import Any, Dict, Iterator, List, Optional

from src.domain.multimodal.image_entities import CrossModalLink, ImageEntity
from src.infrastructure.db.sqlite_client import SQLiteClient

_logger = logging.getLogger("sitrep.repos.multimodal")


class SQLiteMultimodalRepo:
    """Stores images, cross-modal links, and media assets in SQLite."""

    TABLE_IMAGES = "images"
    TABLE_LINKS = "cross_modal_links"
    TABLE_MEDIA = "media_assets"

    def __init__(self, client: SQLiteClient) -> None:
        """Wire the sqlite client and create tables (idempotent)."""
        self.client = client
        self._init_tables()

    def _init_tables(self) -> None:
        """Create the multimodal tables if absent (no migration of existing tables)."""
        with self.client.transaction():
            self.client.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE_IMAGES} ("
                "id TEXT PRIMARY KEY, source TEXT, caption TEXT, embedding BLOB, "
                "linked_passage_ids TEXT, metadata TEXT, created_at TEXT NOT NULL)"
            )
            self.client.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE_LINKS} ("
                "id TEXT PRIMARY KEY, image_id TEXT, passage_id TEXT, kind TEXT, "
                "weight REAL, metadata TEXT, created_at TEXT NOT NULL)"
            )
            self.client.execute(
                f"CREATE INDEX IF NOT EXISTS idx_xmod_link_img ON {self.TABLE_LINKS}(image_id)"
            )
            self.client.execute(
                f"CREATE INDEX IF NOT EXISTS idx_xmod_link_pass ON {self.TABLE_LINKS}(passage_id)"
            )
            self.client.execute(
                f"CREATE TABLE IF NOT EXISTS {self.TABLE_MEDIA} ("
                "id TEXT PRIMARY KEY, kind TEXT, source TEXT, transcript TEXT, "
                "segments TEXT, embedding BLOB, linked_passage_ids TEXT, "
                "metadata TEXT, created_at TEXT NOT NULL)"
            )

    # ----------------------------------------------------------------- images
    def save_image(self, image: ImageEntity) -> str:
        """Insert/replace an image (embedding pickled)."""
        blob = sqlite3.Binary(pickle.dumps(list(image.visual_embedding))) if image.visual_embedding else None
        with self.client.transaction():
            self.client.execute(
                f"INSERT OR REPLACE INTO {self.TABLE_IMAGES} "
                "(id, source, caption, embedding, linked_passage_ids, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    image.id,
                    image.source,
                    image.caption,
                    blob,
                    self.client.dumps_json(image.linked_passage_ids),
                    self.client.dumps_json(image.metadata),
                    image.created_at,
                ),
            )
        return image.id

    def get_image(self, image_id: str) -> Optional[ImageEntity]:
        """Return an image by id (or None)."""
        row = self.client.fetchone(f"SELECT * FROM {self.TABLE_IMAGES} WHERE id=?", (image_id,))
        return self._row_to_image(row) if row else None

    def iter_images_with_embeddings(self) -> Iterator[ImageEntity]:
        """Yield images that have a visual embedding (cross-modal retrieval path)."""
        rows = self.client.fetchall(
            f"SELECT * FROM {self.TABLE_IMAGES} WHERE embedding IS NOT NULL"
        )
        for r in rows:
            yield self._row_to_image(r)

    @staticmethod
    def _row_to_image(row: sqlite3.Row) -> ImageEntity:
        embedding = None
        if row["embedding"] is not None:
            try:
                embedding = pickle.loads(bytes(row["embedding"]))
            except Exception:  # pragma: no cover
                embedding = None
        return ImageEntity(
            id=row["id"],
            source=row["source"] or "",
            caption=row["caption"] or "",
            visual_embedding=embedding,
            linked_passage_ids=SQLiteClient.loads_json(row["linked_passage_ids"], []),
            metadata=SQLiteClient.loads_json(row["metadata"], {}),
            created_at=row["created_at"],
        )

    # ----------------------------------------------------------------- links
    def save_link(self, link: CrossModalLink) -> str:
        """Insert/replace a cross-modal link."""
        with self.client.transaction():
            self.client.execute(
                f"INSERT OR REPLACE INTO {self.TABLE_LINKS} "
                "(id, image_id, passage_id, kind, weight, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    link.id,
                    link.image_id,
                    link.passage_id,
                    link.kind,
                    link.weight,
                    self.client.dumps_json(link.metadata),
                    link.created_at,
                ),
            )
        return link.id

    def links_for_image(self, image_id: str) -> List[CrossModalLink]:
        """Return cross-modal links from *image_id*."""
        rows = self.client.fetchall(
            f"SELECT * FROM {self.TABLE_LINKS} WHERE image_id=?", (image_id,)
        )
        return [self._row_to_link(r) for r in rows]

    def links_for_passage(self, passage_id: str) -> List[CrossModalLink]:
        """Return cross-modal links into *passage_id* (passage → images)."""
        rows = self.client.fetchall(
            f"SELECT * FROM {self.TABLE_LINKS} WHERE passage_id=?", (passage_id,)
        )
        return [self._row_to_link(r) for r in rows]

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> CrossModalLink:
        return CrossModalLink(
            id=row["id"],
            image_id=row["image_id"],
            passage_id=row["passage_id"],
            kind=row["kind"],
            weight=row["weight"],
            metadata=SQLiteClient.loads_json(row["metadata"], {}),
            created_at=row["created_at"],
        )

    # ----------------------------------------------------------------- media (audio/video)
    def save_media(self, entity: Any) -> str:
        """Persist an AudioEntity/VideoEntity as a media asset row."""
        emb = getattr(entity, "embedding", None) or getattr(entity, "visual_embedding", None)
        blob = sqlite3.Binary(pickle.dumps(list(emb))) if emb else None
        kind = "video" if isinstance(entity, str) else entity.__class__.__name__.lower().replace("entity", "")
        segments = [s.to_dict() if hasattr(s, "to_dict") else s for s in getattr(entity, "segments", [])]
        with self.client.transaction():
            self.client.execute(
                f"INSERT OR REPLACE INTO {self.TABLE_MEDIA} "
                "(id, kind, source, transcript, segments, embedding, linked_passage_ids, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    entity.id,
                    kind,
                    getattr(entity, "source", ""),
                    getattr(entity, "transcript", ""),
                    self.client.dumps_json(segments),
                    blob,
                    self.client.dumps_json(getattr(entity, "linked_passage_ids", [])),
                    self.client.dumps_json(getattr(entity, "metadata", {})),
                    entity.created_at,
                ),
            )
        return entity.id

    def count_media(self, kind: Optional[str] = None) -> int:
        """Return the number of media assets (optionally filtered by kind)."""
        if kind:
            row = self.client.fetchone(
                f"SELECT COUNT(*) AS c FROM {self.TABLE_MEDIA} WHERE kind=?", (kind,)
            )
        else:
            row = self.client.fetchone(f"SELECT COUNT(*) AS c FROM {self.TABLE_MEDIA}")
        return int(row["c"]) if row else 0
