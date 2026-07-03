"""Personalized PageRank (PPR) — the graph-importance signal (Quantico Pattern 4).

Power iteration with teleportation and hub suppression, operating on a sparse
adjacency dict. The caller supplies the subgraph (typically the candidate
passages plus their similarity edges) and the seed nodes (query entities / top
candidates). Pure Python, no dependencies.

Implements (verbatim from Quantico):
    p_{t+1} = alpha * (p_t . T) + (1 - alpha) * s
    alpha = 0.85 (damping), gamma = 0.8 (hub suppression), tol = 1e-6
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Sequence

_logger = logging.getLogger("sitrep.retrieval.ppr")

DEFAULT_ALPHA = 0.85
DEFAULT_GAMMA = 0.8
DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 100


class PPREngine:
    """Power-iteration PPR over a sparse adjacency graph."""

    def run_ppr(
        self,
        adjacency: Mapping[str, Mapping[str, float]],
        seed_ids: Sequence[str],
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        tol: float = DEFAULT_TOL,
        max_iter: int = DEFAULT_MAX_ITER,
    ) -> Dict[str, float]:
        """Return PPR scores per node (normalized to sum 1).

        Parameters
        ----------
        adjacency:
            ``node -> {neighbor: weight}`` (non-negative weights).
        seed_ids:
            Teleport seeds (e.g. query entities / top candidates). If empty, the
            teleport vector is uniform over all nodes.
        """
        nodes = set(adjacency.keys())
        for nbrs in adjacency.values():
            nodes.update(nbrs.keys())
        nodes.update(seed_ids)
        if not nodes:
            return {}

        seeds = [s for s in seed_ids if s] or list(nodes)
        s = {sid: 1.0 / len(seeds) for sid in seeds}

        # Out-degree + hub detection (down-weight very high-degree nodes by gamma).
        outdeg = {n: sum(adjacency.get(n, {}).values()) for n in nodes}
        deg_vals = [d for d in outdeg.values() if d > 0]
        median_deg = sorted(deg_vals)[len(deg_vals) // 2] if deg_vals else 0.0

        p = dict(s)
        for _ in range(max(1, int(max_iter))):
            p_new: Dict[str, float] = {n: (1.0 - alpha) * s.get(n, 0.0) for n in nodes}
            dangling_mass = 0.0
            for src, nbrs in adjacency.items():
                outw = sum(nbrs.values())
                if outw <= 0:
                    dangling_mass += alpha * p.get(src, 0.0)
                    continue
                hub = gamma if (median_deg > 0 and outdeg[src] > median_deg * 3) else 1.0
                psrc = alpha * p.get(src, 0.0)
                for dst, w in nbrs.items():
                    p_new[dst] = p_new.get(dst, 0.0) + psrc * (w / outw) * hub
            # Redistribute dangling (dead-end) mass to the seeds (teleport).
            if dangling_mass > 0 and seeds:
                share = dangling_mass / len(seeds)
                for sid in seeds:
                    p_new[sid] = p_new.get(sid, 0.0) + share

            tot = sum(p_new.values()) or 1.0
            for n in p_new:
                p_new[n] /= tot
            diff = sum(abs(p_new[n] - p.get(n, 0.0)) for n in nodes)
            p = p_new
            if diff < tol:
                break
        return p
