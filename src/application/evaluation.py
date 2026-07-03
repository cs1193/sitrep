"""Evaluation harness (E3-lite): retrieval + answer-quality metrics over a labeled set.

Loads a JSONL eval set (``{query, expected_answer, relevant_ids}``) and computes
standard IR metrics (Precision/Recall/MRR/NDCG @K) plus optional answer-similarity.
Used to establish a baseline *before* and measure gains *after* each upgrade
(e.g. Phase A retrieval). Pure Python, no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.evaluation")


@dataclass
class EvalSample:
    """A single labeled evaluation item."""

    query: str
    expected_answer: str = ""
    relevant_ids: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated evaluation metrics."""

    n: int = 0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    answer_similarity: float = 0.0
    answered: int = 0
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "precision@5": round(self.precision_at_5, 4),
            "precision@10": round(self.precision_at_10, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "answer_similarity": round(self.answer_similarity, 4),
            "answered": self.answered,
        }

    def to_table(self) -> str:
        """Return a compact, human-readable metrics table."""
        d = self.to_dict()
        rows = [(k, v) for k, v in d.items() if k not in ("label", "n", "answered")]
        width = max(len(k) for k, _ in rows)
        lines = [f"Eval report — label={self.label or '-'}  n={self.n}  answered={self.answered}"]
        lines.append("  " + "-" * (width + 14))
        for k, v in rows:
            lines.append(f"  {k:<{width}}  {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- IO
def load_eval_jsonl(path) -> List[EvalSample]:
    """Load eval samples from a JSONL file (``query`` required; ``expected_answer``
    and ``relevant_ids`` optional)."""
    samples: List[EvalSample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(
                EvalSample(
                    query=d["query"],
                    expected_answer=d.get("expected_answer", ""),
                    relevant_ids=[str(x) for x in d.get("relevant_ids", [])],
                )
            )
    _logger.info("loaded %d eval samples from %s", len(samples), path)
    return samples


# --------------------------------------------------------------------------- metrics
def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    rel = set(relevant)
    top = list(retrieved)[:k]
    return len([r for r in top if r in rel]) / k


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant ids recovered in the top-k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = list(retrieved)[:k]
    return len([r for r in top if r in rel]) / len(rel)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none)."""
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @k (binary relevance)."""
    rel = set(relevant)
    dcg = 0.0
    for i, r in enumerate(list(retrieved)[:k], start=1):
        if r in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def answer_similarity(answer: str, expected: str, embedder: Any = None) -> float:
    """Similarity between an answer and the expected answer in [0, 1]."""
    if not answer or not expected:
        return 0.0
    if embedder is not None:
        try:
            return max(0.0, cosine_similarity(embedder.embed(answer), embedder.embed(expected)))
        except Exception:  # pragma: no cover
            pass
    a = set(answer.lower().split())
    b = set(expected.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(b)


# --------------------------------------------------------------------------- runner
RetrieveFn = Callable[[str, int], List[str]]
AnswerFn = Callable[[str], str]


def evaluate(
    samples: List[EvalSample],
    retrieve_fn: RetrieveFn,
    answer_fn: Optional[AnswerFn] = None,
    embedder: Any = None,
    top_k: int = 10,
    label: str = "",
) -> EvalReport:
    """Run retrieval (and optional answer) evaluation over *samples*.

    ``retrieve_fn(query, k)`` must return ranked passage ids; ``answer_fn(query)``
    returns the generated answer (or None to skip answer scoring).
    """
    n = len(samples)
    if n == 0:
        return EvalReport(label=label)
    agg = {k: 0.0 for k in (
        "p5", "p10", "r5", "r10", "mrr", "n5", "n10", "ans"
    )}
    answered = 0
    for s in samples:
        retrieved = list(retrieve_fn(s.query, top_k))
        agg["p5"] += precision_at_k(retrieved, s.relevant_ids, 5)
        agg["p10"] += precision_at_k(retrieved, s.relevant_ids, 10)
        agg["r5"] += recall_at_k(retrieved, s.relevant_ids, 5)
        agg["r10"] += recall_at_k(retrieved, s.relevant_ids, 10)
        agg["mrr"] += mrr(retrieved, s.relevant_ids)
        agg["n5"] += ndcg_at_k(retrieved, s.relevant_ids, 5)
        agg["n10"] += ndcg_at_k(retrieved, s.relevant_ids, 10)
        if answer_fn is not None:
            ans = answer_fn(s.query)
            if ans:
                answered += 1
                agg["ans"] += answer_similarity(ans, s.expected_answer, embedder)
    report = EvalReport(
        n=n,
        precision_at_5=agg["p5"] / n,
        precision_at_10=agg["p10"] / n,
        recall_at_5=agg["r5"] / n,
        recall_at_10=agg["r10"] / n,
        mrr=agg["mrr"] / n,
        ndcg_at_5=agg["n5"] / n,
        ndcg_at_10=agg["n10"] / n,
        answer_similarity=(agg["ans"] / n) if answer_fn is not None else 0.0,
        answered=answered,
        label=label,
    )
    return report


__all__ = [
    "EvalSample",
    "EvalReport",
    "load_eval_jsonl",
    "evaluate",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "answer_similarity",
]
