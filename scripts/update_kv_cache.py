#!/usr/bin/env python3
"""Precompute transformer KV caches for all passages (extras ``[llm]`` + torch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Parse args and precompute KV caches."""
    parser = argparse.ArgumentParser(description="Precompute KV caches for SITREP passages.")
    parser.add_argument("--model", default=None, help="HF causal LM to use (defaults to config)")
    parser.add_argument("--limit", type=int, default=None, help="Max passages to cache")
    args = parser.parse_args()

    from src.application import build_application
    from src.infrastructure.kv_cache.precomputer import KVCachePrecomputer

    app = build_application()
    model_name = args.model or app.config.hf_llm_model
    precomputer = KVCachePrecomputer(model_name, app.kv_repo, app.passage_repo)

    if not precomputer.is_available():
        app.close()
        print("transformers/torch not installed. Install with: uv sync --extra llm")
        return

    n = precomputer.precompute(limit=args.limit)
    app.close()
    print(f"\nPrecomputed KV caches for {n} passages (model={model_name}).")


if __name__ == "__main__":
    main()
