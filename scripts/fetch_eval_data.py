#!/usr/bin/env python3
"""Fetch BEIR/SciFact eval data from HuggingFace and build SITREP eval JSONLs.

Regenerates:
  eval/scifact_corpus.jsonl         — full 5,183-doc corpus (passages to ingest)
  eval/scifact_corpus_small.jsonl   — 283 relevant + 1,000 distractors (faster eval)
  eval/scifact_queries.jsonl        — 300 test queries
  eval/scifact_qrels.jsonl          — relevance judgments
  eval/scifact_eval.jsonl           — SITREP format (query, expected_answer, relevant_ids)

Source: https://huggingface.co/datasets/BeIR/scifact (corpus + queries parquet)
        https://huggingface.co/datasets/BeIR/scifact-qrels (test.tsv)

Requires: duckdb (``uv run --with duckdb python scripts/fetch_eval_data.py``)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL = ROOT / "eval"
RAW = EVAL / "raw"

HF = "https://huggingface.co/datasets"
FILES = {
    "corpus.parquet": f"{HF}/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{HF}/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "test.tsv": f"{HF}/BeIR/scifact-qrels/resolve/main/test.tsv",
}
UA = "SITREP-eval/1.0 (+local context-engineering baseline)"


def _download():
    """Download the three source files into eval/raw/."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  exists: {name}")
            continue
        print(f"  downloading: {name}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())


def _s(x):
    """Stringify (normalize int/string id mismatch between qrels and parquet)."""
    return str(x) if x is not None else None


def _build():
    """Parse source files and write SITREP eval JSONLs."""
    import duckdb  # type: ignore

    # Corpus
    ccols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{RAW}/corpus.parquet')").fetchall()]
    cidc = "_id" if "_id" in ccols else ccols[0]
    crows = duckdb.sql(f'SELECT "{cidc}", title, text FROM read_parquet(\'{RAW}/corpus.parquet\')').fetchall()
    corpus = {_s(r[0]): (r[1] or "", r[2] or "") for r in crows}
    print(f"  corpus: {len(corpus)} docs")

    # Queries
    qcols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{RAW}/queries.parquet')").fetchall()]
    qidc = "_id" if "_id" in qcols else qcols[0]
    qtxt = "text" if "text" in qcols else qcols[-1]
    qrows = duckdb.sql(f'SELECT "{qidc}", {qtxt} FROM read_parquet(\'{RAW}/queries.parquet\')').fetchall()
    queries = {_s(r[0]): (r[1] or "") for r in qrows}
    print(f"  queries: {len(queries)}")

    # Qrels
    qrels = []
    with (RAW / "test.tsv").open() as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for r in reader:
            if len(r) >= 3:
                qrels.append({"query_id": _s(r[0]), "corpus_id": _s(r[1]), "score": int(r[2]) if str(r[2]).isdigit() else 1})
    # Filter to self-contained set
    fq = [q for q in qrels if q["corpus_id"] in corpus and q["query_id"] in queries]
    qids = sorted({q["query_id"] for q in fq})
    cids = sorted({q["corpus_id"] for q in fq})
    print(f"  valid qrels: {len(fq)} | queries: {len(qids)} | corpus refs: {len(cids)}")

    # Write corpus.jsonl (full corpus)
    with (EVAL / "scifact_corpus.jsonl").open("w") as f:
        for cid in sorted(corpus):
            t, x = corpus[cid]
            f.write(json.dumps({"passage_id": cid, "document_id": "scifact", "text": (t + ". " + x).strip()}) + "\n")

    # Write corpus_small.jsonl (relevant + 1000 distractors)
    rel_set = set(cids)
    distractors = [c for c in sorted(corpus) if c not in rel_set][:1000]
    with (EVAL / "scifact_corpus_small.jsonl").open("w") as f:
        for cid in cids + distractors:
            t, x = corpus[cid]
            f.write(json.dumps({"passage_id": cid, "document_id": "scifact", "text": (t + ". " + x).strip()}) + "\n")

    # Write queries + qrels + eval.jsonl
    byq = defaultdict(list)
    for q in fq:
        byq[q["query_id"]].append(q["corpus_id"])
    with (EVAL / "scifact_queries.jsonl").open("w") as f:
        for qid in qids:
            f.write(json.dumps({"query_id": qid, "query": queries[qid]}) + "\n")
    with (EVAL / "scifact_qrels.jsonl").open("w") as f:
        for q in fq:
            f.write(json.dumps(q) + "\n")
    with (EVAL / "scifact_eval.jsonl").open("w") as f:
        for qid in qids:
            rel = byq[qid]
            exp = " ".join((corpus[c][0] + ". " + corpus[c][1]).strip() for c in rel)
            f.write(json.dumps({"query": queries[qid], "expected_answer": exp, "relevant_ids": rel}) + "\n")

    print(f"  wrote: corpus ({len(corpus)}), corpus_small ({len(cids)+len(distractors)}), "
          f"queries ({len(qids)}), qrels ({len(fq)}), eval ({len(qids)})")


def main():
    """Download + build all eval JSONLs."""
    print("== fetch BEIR/SciFact eval data ==")
    _download()
    print("== build SITREP eval JSONLs ==")
    _build()
    print(f"== done → {EVAL}/ ==")


if __name__ == "__main__":
    main()
