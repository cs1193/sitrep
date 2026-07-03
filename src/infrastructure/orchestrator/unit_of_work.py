"""Unit of Work: atomic writes across SQLite (+ optional Kuzu/Chroma).

SQLite is the transactional core (BEGIN/COMMIT/ROLLBACK). Non-SQLite operations
(graph/vector) are registered with an optional compensating *undo* and executed
on clean exit; if any fails, prior non-SQLite ops are compensated (best-effort)
and the SQLite transaction is rolled back — so a Kuzu/Chroma failure leaves the
SQLite state unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from src.infrastructure.db.sqlite_client import SQLiteClient

_logger = logging.getLogger("sitrep.orchestrator.uow")


class UnitOfWork:
    """Context manager batching cross-store writes with best-effort atomicity."""

    def __init__(
        self,
        sqlite: SQLiteClient,
        graph_store: Any = None,
        vector_store: Any = None,
    ) -> None:
        """Wire the SQLite client (transactional) and optional derived stores."""
        self.sqlite = sqlite
        self.graph_store = graph_store
        self.vector_store = vector_store
        self._ops: List[Tuple[Callable[[], Any], Optional[Callable[[], Any]]]] = []
        self._done: List[Tuple[Callable[[], Any], Optional[Callable[[], Any]]]] = []
        self._conn: Any = None

    # ----------------------------------------------------------------- context
    def __enter__(self) -> "UnitOfWork":
        """Begin the SQLite transaction."""
        self._conn = self.sqlite.connection
        self._conn.execute("BEGIN")
        self._ops = []
        self._done = []
        return self

    def register(self, do: Callable[[], Any], undo: Optional[Callable[[], Any]] = None) -> None:
        """Register a non-SQLite op with an optional compensating *undo*."""
        self._ops.append((do, undo))

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Commit on success; on any failure roll back SQLite + compensate."""
        if exc_type is not None:
            # Exception escaped the block — roll back everything.
            self._rollback()
            return False
        try:
            for do, undo in self._ops:
                do()
                self._done.append((do, undo))
        except Exception as fail:
            _logger.warning("UnitOfWork non-SQLite op failed (%s); compensating + rolling back", fail)
            self._compensate()
            self._rollback()
            raise
        self._conn.commit()
        _logger.debug("UnitOfWork committed (%d non-SQLite ops)", len(self._done))
        return False

    # ----------------------------------------------------------------- helpers
    def _compensate(self) -> None:
        """Best-effort call of each executed op's undo (reverse order)."""
        for _do, undo in reversed(self._done):
            if undo is None:
                continue
            try:
                undo()
            except Exception as exc:  # pragma: no cover
                _logger.warning("UnitOfWork undo failed: %s", exc)

    def _rollback(self) -> None:
        """Roll back the SQLite transaction (best-effort)."""
        try:
            self._conn.rollback()
        except Exception as exc:  # pragma: no cover
            _logger.warning("UnitOfWork rollback failed: %s", exc)
