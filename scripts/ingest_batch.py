#!/usr/bin/env python3
"""Batch ingest documents from a folder (.txt/.md/.json/.jsonl)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _iter_documents(folder: Path, recursive: bool, exts):
    """Yield ``(document_id, text)`` pairs from files in *folder*."""
    glob = "**/*" if recursive else "*"
    for path in sorted(folder.glob(glob)):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        try:
            yield path.stem, path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover
            logging.warning("could not read %s: %s", path, exc)


def main() -> None:
    """Parse args and run batch ingestion."""
    parser = argparse.ArgumentParser(description="Batch ingest documents into SITREP.")
    parser.add_argument("folder", help="Folder containing documents")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument(
        "--exts",
        default=".txt,.md,.json,.jsonl",
        help="Comma-separated file extensions to ingest",
    )
    args = parser.parse_args()

    from src.application import build_application

    app = build_application()
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    exts = {e if e.startswith(".") else f".{e}" for e in args.exts.split(",")}

    total_passages = total_facts = total_docs = 0
    for doc_id, text in _iter_documents(folder, args.recursive, exts):
        result = app.ingest_uc.execute(text=text, document_id=doc_id)
        total_passages += result.passages
        total_facts += result.facts
        total_docs += 1
        print(
            f"  ✓ {doc_id}: {result.passages} passages, {result.facts} facts "
            f"({result.domain}, {result.method})"
        )
    app.close()
    print(
        f"\nIngested {total_docs} document(s): "
        f"{total_passages} passages, {total_facts} facts."
    )


if __name__ == "__main__":
    main()
