"""Causal reasoning: do-calculus + counterfactuals over a linear SCM (Phase G3).

The engine treats the :class:`~src.domain.causal.CausalGraph` as a linear
structural causal model and implements:

* **Causal effect** of X on Y = sum over directed X→…→Y paths of the product of
  edge weights (the total effect coefficient under intervention ``do(X)``).
* **Backdoor confounders** = common causes of X and Y (variables that are parents
  of X and ancestors of Y) — the set you'd condition on to deconfound.
* **Intervene** ``do(X)`` = the mutilated graph with edges into X removed.
* **Counterfactual** = under the linear SCM, ``Y(do(X=x')) ≈ Y(x) + effect·(x'-x)``.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from src.domain.causal import CausalEdge, CausalGraph

_logger = logging.getLogger("sitrep.reasoning.causal")


class CausalEngine:
    """Do-calculus + counterfactual estimation over a linear SCM."""

    def __init__(self, graph: CausalGraph) -> None:
        """Store the causal graph."""
        self.graph = graph

    # ----------------------------------------------------------------- paths
    def directed_paths(self, source: str, target: str) -> List[List[CausalEdge]]:
        """Return all directed edge-paths from *source* to *target* (cycle-safe)."""
        children = self.graph._children_map()  # noqa: SLF001 (graph helper)
        paths: List[List[CausalEdge]] = []

        def dfs(node: str, acc: List[CausalEdge], visited: Set[str]) -> None:
            if node == target and acc:
                paths.append(list(acc))
                return
            for edge in children.get(node, []):
                if edge.effect in visited:
                    continue  # avoid cycles
                acc.append(edge)
                visited.add(edge.effect)
                dfs(edge.effect, acc, visited)
                acc.pop()
                visited.discard(edge.effect)

        if source == target:
            return []
        dfs(source, [], {source})
        return paths

    def causal_effect(self, treatment: str, outcome: str) -> float:
        """Total causal effect of *treatment* on *outcome* (sum of path products)."""
        if treatment == outcome:
            return 1.0
        total = 0.0
        for path in self.directed_paths(treatment, outcome):
            product = 1.0
            for edge in path:
                product *= edge.weight
            total += product
        return round(total, 6)

    def confidence(self, treatment: str, outcome: str) -> float:
        """Confidence in the effect = min edge confidence along the strongest path."""
        paths = self.directed_paths(treatment, outcome)
        if not paths:
            return 0.0
        best = max(paths, key=lambda p: _path_product(p))
        return min((e.confidence for e in best), default=0.0)

    # ----------------------------------------------------------------- backdoor
    def confounders(self, treatment: str, outcome: str) -> List[str]:
        """Backdoor confounders: parents of *treatment* that are ancestors of *outcome*."""
        if treatment == outcome:
            return []
        parents = set(self.graph.parents(treatment))
        ancestors_of_outcome = self.graph.ancestors(outcome)
        return sorted(parents & ancestors_of_outcome - {treatment})

    def intervene(self, treatment: str) -> CausalGraph:
        """Return the mutilated graph for ``do(treatment)`` (edges into it removed)."""
        mutilated = CausalGraph(variables=list(self.graph.variables))
        mutilated.edges = [e for e in self.graph.edges if e.effect != treatment]
        return mutilated

    # ----------------------------------------------------------------- API
    def estimate_effect(self, treatment: str, outcome: str) -> Dict[str, object]:
        """Return the causal effect plus an explanation (paths + confounders)."""
        effect = self.causal_effect(treatment, outcome)
        paths = self.directed_paths(treatment, outcome)
        return {
            "treatment": treatment,
            "outcome": outcome,
            "effect": effect,
            "confidence": self.confidence(treatment, outcome),
            "n_paths": len(paths),
            "paths": [[e.cause for e in p] + [p[-1].effect] for p in paths],
            "confounders": self.confounders(treatment, outcome),
        }

    def counterfactual(
        self,
        treatment: str,
        outcome: str,
        factual_value: float,
        intervention_value: float,
        factual_outcome: float,
    ) -> Dict[str, object]:
        """Estimate the counterfactual outcome under ``do(treatment = intervention_value)``.

        Linear-SCM approximation: ``Y_cf = Y(x) + effect·(x' - x)``.
        """
        effect = self.causal_effect(treatment, outcome)
        delta = effect * (float(intervention_value) - float(factual_value))
        return {
            "treatment": treatment,
            "outcome": outcome,
            "factual_value": factual_value,
            "intervention_value": intervention_value,
            "factual_outcome": factual_outcome,
            "effect": effect,
            "estimated_outcome": round(float(factual_outcome) + delta, 6),
            "delta": round(delta, 6),
            "confidence": self.confidence(treatment, outcome),
            "explanation": (
                f"Under a linear SCM, changing {treatment} from {factual_value} to "
                f"{intervention_value} shifts {outcome} by effect {effect} * "
                f"({intervention_value} - {factual_value}) = {round(delta, 6)}."
            ),
        }


def _path_product(path: List[CausalEdge]) -> float:
    """Return the product of edge weights along *path*."""
    product = 1.0
    for edge in path:
        product *= edge.weight
    return product
