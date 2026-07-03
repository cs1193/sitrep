"""Causal domain: directed causal graphs over named variables (Phase G3).

A small structural-causal-model vocabulary used by the do-calculus engine —
variables, directed weighted edges (with confidence), and the graph queries the
reasoning layer needs (parents/children/ancestors/descendants).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class CausalVariable:
    """A named node in the causal graph (an observable or latent variable)."""

    name: str
    description: str = ""


@dataclass
class CausalEdge:
    """A directed cause → effect edge with a linear coefficient and confidence."""

    cause: str
    effect: str
    weight: float = 1.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.cause or not self.effect:
            raise ValueError("CausalEdge.cause and CausalEdge.effect are required")
        if abs(self.weight) > 1.0:
            # Not strictly invalid, but coefficients are expected in [-1, 1] for the
            # linear-SCM effect estimates; clamp rather than reject for robustness.
            self.weight = max(-1.0, min(1.0, float(self.weight)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class CausalGraph:
    """A directed acyclic causal graph (the SCM's structure)."""

    variables: List[CausalVariable] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)

    def add_variable(self, name: str, description: str = "") -> CausalVariable:
        """Add (or return) a variable by *name*."""
        existing = self.variable(name)
        if existing is not None:
            return existing
        var = CausalVariable(name=name, description=description)
        self.variables.append(var)
        return var

    def add_edge(self, cause: str, effect: str, weight: float = 1.0, confidence: float = 1.0) -> CausalEdge:
        """Add a directed edge (idempotent on the (cause, effect) pair)."""
        self.add_variable(cause)
        self.add_variable(effect)
        for edge in self.edges:
            if edge.cause == cause and edge.effect == effect:
                edge.weight = weight
                edge.confidence = confidence
                return edge
        edge = CausalEdge(cause=cause, effect=effect, weight=weight, confidence=confidence)
        self.edges.append(edge)
        return edge

    def variable(self, name: str) -> CausalVariable:
        """Return a variable by name (or None)."""
        for var in self.variables:
            if var.name == name:
                return var
        return None  # type: ignore[return-value]

    def _children_map(self) -> Dict[str, List[CausalEdge]]:
        """Return a {cause: [edges]} adjacency."""
        out: Dict[str, List[CausalEdge]] = {v.name: [] for v in self.variables}
        for edge in self.edges:
            out.setdefault(edge.cause, []).append(edge)
        return out

    def _parents_map(self) -> Dict[str, List[CausalEdge]]:
        """Return an {effect: [edges]} adjacency."""
        out: Dict[str, List[CausalEdge]] = {v.name: [] for v in self.variables}
        for edge in self.edges:
            out.setdefault(edge.effect, []).append(edge)
        return out

    def parents(self, name: str) -> List[str]:
        """Return the direct causes (parents) of *name*."""
        return sorted({e.cause for e in self._parents_map().get(name, [])})

    def children(self, name: str) -> List[str]:
        """Return the direct effects (children) of *name*."""
        return sorted({e.effect for e in self._children_map().get(name, [])})

    def ancestors(self, name: str) -> Set[str]:
        """Return all ancestors of *name* (transitive parents)."""
        seen: Set[str] = set()
        stack = list(self.parents(name))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.parents(node))
        return seen

    def descendants(self, name: str) -> Set[str]:
        """Return all descendants of *name* (transitive children)."""
        seen: Set[str] = set()
        stack = list(self.children(name))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.children(node))
        return seen

    def edge(self, cause: str, effect: str) -> CausalEdge:
        """Return the edge (cause→effect) if present (or None)."""
        for edge in self.edges:
            if edge.cause == cause and edge.effect == effect:
                return edge
        return None  # type: ignore[return-value]
