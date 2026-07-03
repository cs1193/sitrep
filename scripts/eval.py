#!/usr/bin/env python3
"""Run the SITREP evaluation harness over a labeled JSONL set (default: BEIR/SciFact).

Ingests a corpus (preserving external ids so retrieval ids match relevance
judgments), then measures Precision/Recall/MRR/NDCG @K (+ optional answer
similarity). Persists a JSON result so 'before' and 'after' runs can be diffed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "eval"


def _ingest_corpus(app, corpus_path: Path) -> int:
    """Ingest passages preserving their external ``passage_id`` (no chunking/extraction).

    Keeping the source id lets retrieved ids match the eval's ``relevant_ids``.
    Embeds in one batch (much faster than per-doc encode with a real model).
    """
    from src.domain.schemas import Passage

    texts: list = []
    ids: list = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["text"])
            ids.append(d["passage_id"])
    if hasattr(app._embedder, "embed_batch"):
        embeddings = app._embedder.embed_batch(texts)
    else:  # pragma: no cover
        embeddings = [app._embedder.embed(t) for t in texts]
    for text, pid, emb in zip(texts, ids, embeddings):
        app.passage_repo.save(
            Passage(text=text, document_id="scifact", id=pid, embedding=emb)
        )
    return len(texts)


def main() -> None:
    """Parse args, ingest, evaluate, print + persist the report."""
    parser = argparse.ArgumentParser(description="Run the SITREP evaluation harness.")
    parser.add_argument("--corpus", default=str(EVAL_DIR / "scifact_corpus.jsonl"))
    parser.add_argument("--eval", default=str(EVAL_DIR / "scifact_eval.jsonl"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of queries")
    parser.add_argument("--label", default="run", help="Label for the result file")
    parser.add_argument("--with-answers", action="store_true", help="Also score answer similarity")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder rerank (isolate fusion)")
    args = parser.parse_args()

    from src.application import build_application
    from src.application.evaluation import evaluate, load_eval_jsonl

    app = build_application()
    if args.no_rerank:
        app._retriever.reranker = None
    try:
        corpus_n = _ingest_corpus(app, Path(args.corpus))
        samples = load_eval_jsonl(args.eval)
        if args.limit:
            samples = samples[: args.limit]
        print(f"Ingested {corpus_n} passages; evaluating {len(samples)} queries (top_k={args.top_k}).")

        retrieve_fn = lambda q, k: [r.passage_id for r in app._retriever.retrieve(q, k)]
        answer_fn = None
        if args.with_answers:
            answer_fn = lambda q: app.query_uc.execute(q).answer

        report = evaluate(
            samples,
            retrieve_fn=retrieve_fn,
            answer_fn=answer_fn,
            embedder=app._embedder,
            top_k=args.top_k,
            label=args.label,
        )
        print(report.to_table())

        results_dir = EVAL_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = results_dir / f"{args.label}_{stamp}.json"
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nResult saved → {out}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
