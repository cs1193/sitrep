#!/usr/bin/env python3
"""Query the SITREP knowledge base from the command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Parse args and run a query."""
    parser = argparse.ArgumentParser(description="Query the SITREP knowledge base.")
    parser.add_argument("query", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve")
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    dto = app.query_uc.execute(args.query, top_k=args.top_k)
    app.close()

    print(f"\n📝 {dto.query}\n")
    print(dto.answer)
    print(
        f"\n— confidence={dto.confidence:.3f} · token_reduction={dto.token_reduction:.3f} "
        f"· backend={dto.backend} · sources={len(dto.results)}"
    )
    if dto.needs_clarification and dto.clarification_question:
        print(f"\n⚠️  Clarification requested: {dto.clarification_question}")
    print("\nSources:")
    for r in dto.results:
        print(f"  [{r.final_score:.3f}] {r.text[:120]}")


if __name__ == "__main__":
    main()
