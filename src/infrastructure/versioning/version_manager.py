"""Version manager: tar.gz snapshots of ``.sitrep/`` with restore + rollback.

Snapshots are stored in a sibling directory (``<name>_snapshots``) outside the
snapshotted tree to avoid recursion. Uses only the stdlib (``tarfile``/``shutil``).
"""
from __future__ import annotations

import logging
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

_logger = logging.getLogger("sitrep.versioning")


class VersionManager:
    """Snapshot/restore manager for the SITREP data directory."""

    def __init__(self, base_dir: Union[str, Path]) -> None:
        """Configure the data directory and its sibling snapshots directory."""
        self.base_dir = Path(base_dir)
        self.snapshots_dir = self.base_dir.parent / f"{self.base_dir.name}_snapshots"

    # ----------------------------------------------------------------- snapshot
    def snapshot(self, label: Optional[str] = None) -> Path:
        """Create a gzipped tarball of ``base_dir`` and return its path."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_label = (label or "snapshot").strip().replace(" ", "_") or "snapshot"
        name = f"{safe_label}_{stamp}.tar.gz"
        out = self.snapshots_dir / name

        def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            # Skip caches and WAL/SHM sidecars that are unsafe to restore.
            if "__pycache__" in info.name:
                return None
            if info.name.endswith(("-wal", "-shm")):
                return None
            return info

        with tarfile.open(out, "w:gz") as tar:
            tar.add(self.base_dir, arcname=self.base_dir.name, filter=_filter)
        _logger.info("snapshot created: %s", out)
        return out

    # ----------------------------------------------------------------- list
    def list_snapshots(self) -> List[dict]:
        """Return metadata for each snapshot (name, path, size, mtime)."""
        if not self.snapshots_dir.exists():
            return []
        items = []
        for p in sorted(self.snapshots_dir.glob("*.tar.gz")):
            st = p.stat()
            items.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "size_mb": round(st.st_size / (1024 * 1024), 3),
                    "created_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                }
            )
        return items

    # ----------------------------------------------------------------- restore
    def restore(self, name: str, backup_current: bool = True) -> Path:
        """Restore snapshot *name* over ``base_dir`` (backing up the current state)."""
        snapshot = self._resolve(name)
        if snapshot is None:
            raise FileNotFoundError(f"snapshot not found: {name}")
        if backup_current and self.base_dir.exists():
            bak = self.base_dir.with_name(f"{self.base_dir.name}_restored_bak")
            if bak.exists():
                shutil.rmtree(bak)
            shutil.move(str(self.base_dir), str(bak))
            _logger.info("current data backed up to %s", bak)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(snapshot, "r:gz") as tar:
            tar.extractall(self.base_dir.parent)  # arcname == base_dir.name
        _logger.info("restored %s → %s", snapshot.name, self.base_dir)
        return self.base_dir

    # ----------------------------------------------------------------- delete
    def delete(self, name: str) -> bool:
        """Delete snapshot *name*; return True if removed."""
        snapshot = self._resolve(name)
        if snapshot is None:
            return False
        snapshot.unlink()
        _logger.info("deleted snapshot %s", snapshot.name)
        return True

    # ----------------------------------------------------------------- helpers
    def _resolve(self, name: str) -> Optional[Path]:
        """Resolve a snapshot by exact name or label prefix."""
        candidate = self.snapshots_dir / name
        if candidate.exists():
            return candidate
        matches = sorted(self.snapshots_dir.glob(f"{name}*.tar.gz"))
        return matches[-1] if matches else None
