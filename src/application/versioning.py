"""Versioning use case: snapshot/list/restore the ``.sitrep/`` directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from src.application.dto import VersionDTO
from src.domain.schemas import Decision
from src.infrastructure.lineage import LineageTracker
from src.infrastructure.versioning import VersionManager
from src.utils.common import utc_now_iso
from src.utils.constants import DEC_VERSION
from src.utils.decorators import log_execution

_logger = logging.getLogger("sitrep.usecase.versioning")


class VersionManagerUseCase:
    """Wraps :class:`VersionManager` with lineage recording."""

    def __init__(self, version_manager: VersionManager, lineage_tracker: LineageTracker) -> None:
        """Wire the version manager and lineage tracker."""
        self.version_manager = version_manager
        self.lineage_tracker = lineage_tracker

    @log_execution
    def snapshot(self, label: Optional[str] = None) -> VersionDTO:
        """Create a snapshot and return its metadata."""
        path = self.version_manager.snapshot(label)
        stat = path.stat()
        dto = VersionDTO(
            name=path.name,
            path=str(path),
            size_mb=round(stat.st_size / (1024 * 1024), 3),
            created_at=utc_now_iso(),
        )
        self._record("snapshot", {"snapshot": path.name, "size_mb": dto.size_mb})
        return dto

    def list_snapshots(self) -> List[VersionDTO]:
        """Return all stored snapshots as DTOs."""
        return [VersionDTO.from_dict(s) for s in self.version_manager.list_snapshots()]

    @log_execution
    def restore(self, name: str) -> str:
        """Restore snapshot *name* (backing up the current state)."""
        self.version_manager.restore(name)
        self._record("restore", {"snapshot": name})
        return name

    @log_execution
    def delete(self, name: str) -> bool:
        """Delete snapshot *name*; return True if removed."""
        removed = self.version_manager.delete(name)
        self._record("delete", {"snapshot": name, "removed": removed})
        return removed

    # ----------------------------------------------------------------- helpers
    def _record(self, action: str, outputs: dict) -> None:
        """Record a versioning decision in the lineage."""
        self.lineage_tracker.record(
            Decision(
                agent_id="version",
                decision_type=DEC_VERSION,
                action=action,
                inputs={},
                outputs=outputs,
                rationale="version management",
            )
        )
