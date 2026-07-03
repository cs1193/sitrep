#!/usr/bin/env python3
"""Run the memory-hygiene forgetting pass (Phase B).

Defaults to a non-destructive DRY RUN. Pass ``--apply`` to actually mutate
memory_status (still never hard-deletes — PERMANENTLY_DELETED is never chosen by
the default strategy mapping).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Parse args and run the forgetting pass."""
    parser = argparse.ArgumentParser(description="Run SITREP memory-hygiene forgetting.")
    parser.add_argument("--apply", action="store_true", help="Actually mutate (default is dry-run)")
    parser.add_argument("--decay-only", action="store_true", help="Only apply the daily importance decay")
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    try:
        if args.decay_only:
            report = app.forgetting_uc.decay_all(dry_run=not args.apply)
        else:
            report = app.forgetting_uc.execute(dry_run=not args.apply)
        print(json.dumps(report, indent=2, default=str))
    finally:
        app.close()


if __name__ == "__main__":
    main()
