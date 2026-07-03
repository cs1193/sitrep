"""ChromaDB vector store client (lazy).

Stores embeddings for passages, facts, and schemas in a persistent on-disk
client at ``.sitrep/vectors``. ``chromadb`` is optional; callers should handle
``ImportError`` and fall back to the SQLite FTS path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_logger = logging.getLogger("sitrep.db.chroma")


class ChromaClient:
    """Lazy ChromaDB persistent client with collection helpers."""

    def __init__(self, persist_dir: Union[str, Path]) -> None:
        """Store the persistence directory; the client opens lazily on first use."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        self._collections: Dict[str, Any] = {}

    # ----------------------------------------------------------------- lifecycle
    def _ensure_open(self) -> None:
        """Open the persistent Chroma client (imports ``chromadb`` lazily)."""
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "chromadb is not installed. Install with: uv sync --extra rag"
            ) from exc
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        _logger.info("chroma persistent client opened at %s", self.persist_dir)

    def collection(self, name: str):
        """Return (creating if needed) the named collection."""
        self._ensure_open()
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name=name)
        return self._collections[name]

    # ----------------------------------------------------------------- operations
    def add(
        self,
        collection: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Add (or upsert) vectors into *collection*."""
        if not ids:
            return
        coll = self.collection(collection)
        coll.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=list(metadatas) if metadatas else None,
        )

    def query(
        self,
        collection: str,
        embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Query *collection* and return ``(id, distance, document, metadata)`` tuples.

        Distance is converted to a similarity score in [0, 1] (``1 - distance/2``).
        """
        coll = self.collection(collection)
        result = coll.query(
            query_embeddings=[list(embedding)],
            n_results=int(top_k),
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        out: List[Tuple[str, float, str, Dict[str, Any]]] = []
        if not result or not result.get("ids"):
            return out
        ids = result["ids"][0]
        dists = result.get("distances", [[0.0] * len(ids)])[0]
        docs = result.get("documents", [[""] * len(ids)])[0]
        metas = result.get("metadatas", [[{}] * len(ids)])[0]
        for i, _id in enumerate(ids):
            sim = max(0.0, 1.0 - float(dists[i]) / 2.0)
            out.append((str(_id), sim, str(docs[i]), dict(metas[i] or {})))
        return out

    def delete(self, collection: str, ids: Sequence[str]) -> None:
        """Delete vectors by *ids* from *collection*."""
        if not ids:
            return
        self.collection(collection).delete(ids=list(ids))

    def count(self, collection: str) -> int:
        """Return the number of items in *collection*."""
        try:
            return int(self.collection(collection).count())
        except Exception:  # pragma: no cover
            return 0

    def close(self) -> None:
        """Drop cached collections."""
        self._collections.clear()
        self._client = None

    def __enter__(self) -> "ChromaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
