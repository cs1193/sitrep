#!/usr/bin/env python3
"""Snapshot, list, restore, or delete SITREP data snapshots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Dispatch snapshot/list/restore/delete actions."""
    parser = argparse.ArgumentParser(description="Manage SITREP snapshots.")
    parser.add_argument("action", choices=["snapshot", "list", "restore", "delete"])
    parser.add_argument("--label", default=None, help="Snapshot label (for 'snapshot')")
    parser.add_argument("--name", default=None, help="Snapshot name (for 'restore'/'delete')")
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    try:
        if args.action == "snapshot":
            dto = app.version_uc.snapshot(args.label)
            print(f"Snapshot created: {dto.name} ({dto.size_mb} MB) -> {dto.path}")
        elif args.action == "list":
            snapshots = app.version_uc.list_snapshots()
            if not snapshots:
                print("No snapshots found.")
            for s in snapshots:
                print(f"  {s.name}  {s.size_mb} MB  {s.created_at}")
        elif args.action == "restore":
            if not args.name:
                print("--name is required for restore")
                return
            app.version_uc.restore(args.name)
            print(f"Restored: {args.name}")
        elif args.action == "delete":
            if not args.name:
                print("--name is required for delete")
                return
            removed = app.version_uc.delete(args.name)
            print(f"Deleted: {args.name}" if removed else f"Not found: {args.name}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
