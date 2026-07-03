#!/usr/bin/env python3
"""Inspect SITREP lineage: trace a decision or list recent decisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Parse args and print lineage information."""
    parser = argparse.ArgumentParser(description="Analyze SITREP lineage.")
    parser.add_argument("--decision-id", default=None, help="Trace a specific decision")
    parser.add_argument("--recent", type=int, default=0, help="Show the N most recent decisions")
    parser.add_argument("--episode", default=None, help="Filter decisions by episode id")
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    try:
        if args.decision_id:
            trace = app.lineage_uc.trace(args.decision_id)
            print(json.dumps(trace.to_dict(), indent=2, default=str))
        elif args.episode:
            decisions = app.lineage_uc.by_episode(args.episode)
            print(json.dumps(decisions, indent=2, default=str))
        else:
            limit = args.recent or 20
            decisions = app.lineage_uc.recent(limit)
            print(f"Recent {len(decisions)} decisions:")
            for d in decisions:
                print(f"  [{d.get('timestamp', '')}] {d.get('decision_type')} :: {d.get('action')}")
                print(f"      id={d.get('id')}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
