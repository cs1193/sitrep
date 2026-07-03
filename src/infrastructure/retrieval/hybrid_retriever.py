"""Hybrid retriever with learnable weighted fusion.

Fuses three signals — BM25/FTS5, dense-vector, and graph proximity — with
weights that are updated online (SGD) and in batch (least squares) from user
feedback. Reranks the top candidates with a cross-encoder when available.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.domain.interfaces import (
    EmbeddingGateway,
    FactRepository,
    GraphStore,
    PassageRepository,
    Retriever,
    Reranker,
    VectorStore,
)
from src.domain.schemas import Passage
from src.domain.value_objects import RetrievalResult
from src.infrastructure.retrieval.ppr import PPREngine
from src.infrastructure.retrieval.temporal_scorer import TemporalScorer
from src.utils.constants import COLL_PASSAGES

_logger = logging.getLogger("sitrep.retrieval.hybrid")


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a score dict to [0, 1] (empty → empty, flat → all 1.0)."""
    if not scores:
        return {}
    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    rng = mx - mn
    return {k: (v - mn) / rng for k, v in scores.items()}


class WeightedFusion:
    """Maintains (bm25, vector, graph) weights and fuses per-item scores."""

    def __init__(self, weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3), lr: float = 0.05) -> None:
        """Initialize normalized weights and the online learning rate."""
        self.bm25_w, self.vector_w, self.graph_w = self._normalize(weights)
        self.lr = lr

    @staticmethod
    def _normalize(weights: Sequence[float]) -> Tuple[float, float, float]:
        w = [max(0.0, float(x)) for x in weights]
        if len(w) != 3:
            raise ValueError("expected exactly 3 weights (bm25, vector, graph)")
        total = sum(w) or 1.0
        return (w[0] / total, w[1] / total, w[2] / total)

    @property
    def weights(self) -> Tuple[float, float, float]:
        """Return current weights as a tuple."""
        return (self.bm25_w, self.vector_w, self.graph_w)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Replace weights (re-normalized)."""
        self.bm25_w, self.vector_w, self.graph_w = self._normalize(weights)

    def fuse(
        self,
        bm25: Dict[str, float],
        vector: Dict[str, float],
        graph: Dict[str, float],
    ) -> Dict[str, float]:
        """Fuse three score dicts into a single score per item id."""
        ids = set(bm25) | set(vector) | set(graph)
        return {
            i: self.bm25_w * bm25.get(i, 0.0)
            + self.vector_w * vector.get(i, 0.0)
            + self.graph_w * graph.get(i, 0.0)
            for i in ids
        }

    def update_online(
        self,
        feedback_rows: Iterable[Dict[str, Tuple[float, float, float, float]]],
        lr: Optional[float] = None,
    ) -> None:
        """SGD update: nudge weights toward channels correlated with feedback.

        Each row maps item_id → (bm25, vector, graph, feedback) where feedback
        is typically in {-1, 0, +1}.
        """
        rate = self.lr if lr is None else lr
        grad = [0.0, 0.0, 0.0]
        for row in feedback_rows:
            for _id, (b, v, g, fb) in row.items():
                grad[0] += fb * b
                grad[1] += fb * v
                grad[2] += fb * g
        self.bm25_w = max(0.0, self.bm25_w + rate * grad[0])
        self.vector_w = max(0.0, self.vector_w + rate * grad[1])
        self.graph_w = max(0.0, self.graph_w + rate * grad[2])
        self.bm25_w, self.vector_w, self.graph_w = self._normalize(
            (self.bm25_w, self.vector_w, self.graph_w)
        )

    def fit_batch(self, rows: Sequence[Tuple[Sequence[float], float]]) -> None:
        """Batch least-squares fit of weights from (features, feedback) rows."""
        if len(rows) < 3:
            _logger.debug("fit_batch needs >=3 rows; skipping")
            return
        try:
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover
            return
        A = np.array([r[0] for r in rows], dtype="float64")
        b = np.array([r[1] for r in rows], dtype="float64")
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception as exc:  # pragma: no cover
            _logger.warning("lstsq failed: %s", exc)
            return
        coef = np.clip(coef, 0.0, None)
        if coef.sum() <= 0:
            return
        self.set_weights(coef.tolist())


class HybridRetriever(Retriever):
    """Fused BM25 + vector + graph retriever with optional reranking."""

    def __init__(
        self,
        passage_repo: PassageRepository,
        fact_repo: Optional[FactRepository] = None,
        embedder: Optional[EmbeddingGateway] = None,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        reranker: Optional[Reranker] = None,
        weights: Optional[Sequence[float]] = None,
        sqlite: Any = None,
        top_k: int = 5,
        ppr_engine: Optional[PPREngine] = None,
        temporal_scorer: Optional[TemporalScorer] = None,
        ppr_alpha: float = 0.85,
        ppr_gamma: float = 0.8,
        ppr_max_iter: int = 100,
        ppr_tol: float = 1e-6,
        ppr_weight: float = 0.0,
        density_weight: float = 0.0,
        bridge_theta: float = 0.3,
        bridge_degree: int = 5,
        temporal_weight: float = 0.0,
        entity_graph_provider: Any = None,
        track_access: bool = False,
    ) -> None:
        """Wire dependencies and load (or default) fusion weights.

        Phase A knobs (``ppr_weight``/``density_weight``/``temporal_weight``)
        default to 0 so the retriever behaves exactly as before unless explicitly
        enabled — keeping all existing callers/tests green.
        """
        self.passage_repo = passage_repo
        self.fact_repo = fact_repo
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker
        self.sqlite = sqlite
        self.top_k = top_k
        # Phase A: graph-importance (PPR), temporal decay, entity density.
        self.ppr_engine = ppr_engine
        self.temporal_scorer = temporal_scorer
        self.ppr_alpha = ppr_alpha
        self.ppr_gamma = ppr_gamma
        self.ppr_max_iter = ppr_max_iter
        self.ppr_tol = ppr_tol
        self.ppr_weight = ppr_weight
        self.density_weight = density_weight
        self.bridge_theta = bridge_theta
        self.bridge_degree = bridge_degree
        self.temporal_weight = temporal_weight
        self.entity_graph_provider = entity_graph_provider
        self.track_access = track_access
        self._entity_adjacency_cache: Optional[Dict[str, Dict[str, float]]] = None
        loaded = weights
        if loaded is None and sqlite is not None:
            try:
                loaded = sqlite.get_fusion_weights()
            except Exception:  # pragma: no cover
                loaded = None
        self.fusion = WeightedFusion(loaded or (1 / 3, 1 / 3, 1 / 3))

    # ----------------------------------------------------------------- retrieval
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Run hybrid retrieval for *query* and return ranked results.

        Phase A: after the learned BM25/vector/graph fusion, applies a temporal
        blend on the semantic channel (``t_w``), then adds PPR graph-importance
        and entity-density boosts. PPR/density/temporal scores are also stashed
        in each result's ``metadata`` for explainability.
        """
        top_k = top_k or self.top_k
        if self.embedder is not None:
            query_emb = self.embedder.embed(query)
        else:
            from src.utils.embedding import hash_embedding

            query_emb = hash_embedding(query)

        passages: Dict[str, Passage] = {}
        bm25_scores = self._bm25(query, passages, top_k)
        vector_scores = self._vector(query_emb, passages, top_k)
        graph_scores = self._graph(query, passages)

        bm25_n = _normalize_scores(bm25_scores)
        vector_n = _normalize_scores(vector_scores)
        graph_n = _normalize_scores(graph_scores)

        # Temporal blend on the semantic (vector) channel (Quantico t_w=0.3).
        temporal_scores = self._temporal(passages)
        if self.temporal_scorer is not None and temporal_scores and self.temporal_weight > 0:
            temporal_n = _normalize_scores(temporal_scores)
            vector_n = self.temporal_scorer.blend(temporal_n, vector_n, self.temporal_weight)

        fused = self.fusion.fuse(bm25_n, vector_n, graph_n)

        # PPR graph-importance + entity-density additive boosts (Phase A).
        density_scores = self._density(query, passages)
        ppr_scores = self._ppr(query, passages, fused)
        ppr_n = _normalize_scores(ppr_scores)
        dens_n = _normalize_scores(density_scores)
        combined = {
            pid: fused.get(pid, 0.0)
            + self.ppr_weight * ppr_n.get(pid, 0.0)
            + self.density_weight * dens_n.get(pid, 0.0)
            for pid in fused
        }

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        rerank_map = self._rerank(query, ranked, passages)

        results: List[RetrievalResult] = []
        for pid, score in ranked:
            p = passages.get(pid)
            if p is None:
                continue
            meta = dict(p.metadata)
            meta["ppr_score"] = round(ppr_n.get(pid, 0.0), 4)
            meta["entity_density"] = round(dens_n.get(pid, 0.0), 4)
            meta["temporal_score"] = round(temporal_scores.get(pid, 0.0), 4)
            results.append(
                RetrievalResult(
                    passage_id=pid,
                    text=p.text,
                    score=score,
                    bm25_score=bm25_scores.get(pid, 0.0),
                    vector_score=vector_scores.get(pid, 0.0),
                    graph_score=graph_scores.get(pid, 0.0),
                    rerank_score=rerank_map.get(pid),
                    metadata=meta,
                )
            )
        if self.track_access and self.passage_repo is not None and results:
            self._bump_access([r.passage_id for r in results])
        return results

    # ----------------------------------------------------------------- channels
    def _bm25(self, query: str, passages: Dict[str, Passage], top_k: int) -> Dict[str, float]:
        """FTS5/BM25 channel; populates *passages* with hits."""
        scores: Dict[str, float] = {}
        if self.passage_repo is None:
            return scores
        try:
            for passage, score in self.passage_repo.search_fts(query, limit=top_k * 3):
                passages[passage.id] = passage
                scores[passage.id] = float(score)
        except Exception as exc:  # pragma: no cover
            _logger.warning("bm25 channel failed: %s", exc)
        return scores

    def _vector(
        self, query_emb: Sequence[float], passages: Dict[str, Passage], top_k: int
    ) -> Dict[str, float]:
        """Dense-vector channel via ChromaDB or a stored-embedding scan fallback."""
        scores: Dict[str, float] = {}
        if self.vector_store is not None:
            try:
                for pid, sim, text, meta in self.vector_store.query(
                    COLL_PASSAGES, query_emb, top_k=top_k * 3
                ):
                    scores[pid] = float(sim)
                    if pid not in passages:
                        passages[pid] = Passage(
                            text=text, document_id=str(meta.get("document_id", "?")), id=pid
                        )
                return scores
            except Exception as exc:  # pragma: no cover
                _logger.warning("vector store query failed: %s", exc)
        # Fallback: cosine over passages that have a stored embedding.
        from src.utils.common import cosine_similarity

        for passage in self._iter_passages_with_embeddings():
            if passage.embedding is None:
                continue
            sim = cosine_similarity(query_emb, passage.embedding)
            if sim > 0.0:
                scores[passage.id] = float(sim)
                passages.setdefault(passage.id, passage)
        return scores

    def _graph(self, query: str, passages: Dict[str, Passage]) -> Dict[str, float]:
        """Graph-proximity channel: boost passages linked to matching facts."""
        scores: Dict[str, float] = {}
        if self.fact_repo is None:
            return scores
        try:
            for fact in self.fact_repo.search(query):
                boost = float(fact.confidence)
                for pid in fact.source_passage_ids:
                    scores[pid] = scores.get(pid, 0.0) + boost
                # expand one hop through the graph if available
                if self.graph_store is not None:
                    for nb in self.graph_store.neighbors("Fact", fact.id, limit=5):
                        nb_pid = nb.get("id")
                        if nb_pid:
                            scores[nb_pid] = scores.get(nb_pid, 0.0) + 0.25 * boost
        except Exception as exc:  # pragma: no cover
            _logger.debug("graph channel failed: %s", exc)
        return scores

    def _temporal(self, passages: Dict[str, Passage]) -> Dict[str, float]:
        """Return temporal-decay scores per passage id (Phase A)."""
        if self.temporal_scorer is None or self.temporal_weight <= 0 or not passages:
            return {}
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        out: Dict[str, float] = {}
        for pid, p in passages.items():
            try:
                created = datetime.fromisoformat(p.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = (now - created).total_seconds() / 86400.0
                out[pid] = self.temporal_scorer.decay(age)
            except Exception:
                out[pid] = self.temporal_scorer.decay(0.0)
        return out

    def _density(self, query: str, passages: Dict[str, Passage]) -> Dict[str, float]:
        """Return normalized query-entity-density scores per passage id (Phase A)."""
        if self.density_weight <= 0 or not passages:
            return {}
        qterms = {t for t in query.lower().split() if len(t) > 3}
        if not qterms:
            return {}
        out: Dict[str, float] = {}
        for pid, p in passages.items():
            pterms = {t for t in p.text.lower().split() if len(t) > 3}
            out[pid] = len(qterms & pterms) / len(qterms)
        return out

    def _ppr(self, query: str, passages: Dict[str, Passage], fused: Dict[str, float]) -> Dict[str, float]:
        """Run PPR over a candidate similarity subgraph; return scores per id (Phase A)."""
        if self.ppr_engine is None or self.ppr_weight <= 0 or len(passages) < 2:
            return {}
        from src.utils.common import cosine_similarity

        # Restrict PPR to the TOP fused candidates — bounds the O(n^2) graph
        # build so cost stays constant regardless of corpus size.
        top_n = min(30, max(self.top_k * 3, 10))
        ranked_ids = [
            pid
            for pid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        ]
        embs: Dict[str, Sequence[float]] = {}
        for pid in ranked_ids:
            p = passages.get(pid)
            if p is None:
                continue
            if p.embedding is not None:
                embs[pid] = p.embedding
            elif self.embedder is not None:
                embs[pid] = self.embedder.embed(p.text)
        ids = list(embs.keys())
        adjacency = self._candidate_adjacency(ids, embs)
        seeds = ids[: max(3, len(ids) // 3)] or ids
        return self.ppr_engine.run_ppr(
            adjacency,
            seeds,
            alpha=self.ppr_alpha,
            gamma=self.ppr_gamma,
            tol=self.ppr_tol,
            max_iter=self.ppr_max_iter,
        )

    def _entity_adjacency(self) -> Dict[str, Dict[str, float]]:
        """Lazily build and cache passage adjacency from shared fact entities."""
        if self._entity_adjacency_cache is None and self.entity_graph_provider is not None:
            try:
                self._entity_adjacency_cache = self.entity_graph_provider() or {}
            except Exception as exc:  # pragma: no cover
                _logger.debug("entity graph build failed: %s", exc)
                self._entity_adjacency_cache = {}
        return self._entity_adjacency_cache or {}

    def _candidate_adjacency(self, ids, embs) -> Dict[str, Dict[str, float]]:
        """Prefer entity-graph edges among candidates; fall back to similarity."""
        adjacency: Dict[str, Dict[str, float]] = {pid: {} for pid in ids}
        entity_adj = self._entity_adjacency()
        if entity_adj:
            idset = set(ids)
            for pid in ids:
                for nbr, w in entity_adj.get(pid, {}).items():
                    if nbr in idset and nbr != pid:
                        adjacency[pid][nbr] = float(w)
        if not any(adjacency.values()):
            from src.utils.common import cosine_similarity

            for a in ids:
                sims = [(b, cosine_similarity(embs[a], embs[b])) for b in ids if b != a]
                sims.sort(key=lambda x: x[1], reverse=True)
                for b, s in sims[: self.bridge_degree]:
                    if s >= self.bridge_theta:
                        adjacency[a][b] = float(s)
        return adjacency

    def _bump_access(self, passage_ids: Sequence[str]) -> None:
        """Increment access_count / last_accessed_at for retrieved passages."""
        if self.passage_repo is None:
            return
        from src.utils.common import utc_now_iso

        for pid in passage_ids:
            passage = self.passage_repo.get(pid)
            if passage is None:
                continue
            passage.metadata["access_count"] = int(passage.metadata.get("access_count", 0)) + 1
            passage.metadata["last_accessed_at"] = utc_now_iso()
            try:
                self.passage_repo.save(passage)
            except Exception:  # pragma: no cover
                pass

    def _rerank(
        self, query: str, ranked: List[Tuple[str, float]], passages: Dict[str, Passage]
    ) -> Dict[str, float]:
        """Rerank the top candidates; return id → rerank score."""
        if self.reranker is None or not ranked:
            return {}
        ids = [pid for pid, _ in ranked]
        docs = [passages[pid].text for pid in ids if pid in passages]
        if not docs:
            return {}
        out: Dict[str, float] = {}
        for idx, score in self.reranker.rerank(query, docs):
            if idx < len(ids):
                out[ids[idx]] = float(score)
        return out

    def _iter_passages_with_embeddings(self) -> Iterable[Passage]:
        """Yield passages that have a stored embedding (repo hook)."""
        if self.passage_repo is None:
            return []
        getter = getattr(self.passage_repo, "iter_with_embeddings", None)
        if getter is None:
            return []
        return getter()

    # ----------------------------------------------------------------- learning
    def update_weights(self, weights: Sequence[float]) -> None:
        """Update fusion weights and persist them (when a sqlite handle is wired)."""
        self.fusion.set_weights(weights)
        if self.sqlite is not None:
            try:
                self.sqlite.seed_fusion_weights(self.fusion.weights)
            except Exception:  # pragma: no cover
                pass

    @property
    def weights(self) -> Tuple[float, float, float]:
        """Return current fusion weights."""
        return self.fusion.weights
