"""Entity-graph builder: passage adjacency from shared fact entities.

This is the "bridge discovery" that gives PPR a *semantic* graph to traverse
(Quantico Patterns 2 + 4). Two passages are linked when they share a fact entity
(subject/object); edge weight grows with the number of shared entities. Pure
function over a fact iterable — no persistence.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable

from src.domain.schemas import Fact

_logger = logging.getLogger("sitrep.retrieval.entity_graph")


class EntityGraphBuilder:
    """Builds a passage-passage adjacency graph from shared fact entities."""

    @staticmethod
    def build(facts: Iterable[Fact]) -> Dict[str, Dict[str, float]]:
        """Return ``{passage_id: {other_passage_id: weight}}``.

        Each shared entity between two passages contributes +1.0 to their edge
        weight (in both directions). Passages with no extracted facts are absent.
        """
        entity_to_passages: Dict[str, set] = defaultdict(set)
        for fact in facts:
            pids = [pid for pid in fact.source_passage_ids if pid]
            if not pids:
                continue
            for entity in {fact.subject.lower(), fact.object_value.lower()}:
                if entity:
                    for pid in pids:
                        entity_to_passages[entity].add(pid)

        adjacency: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for passage_set in entity_to_passages.values():
            passages = sorted(passage_set)
            for i in range(len(passages)):
                for j in range(i + 1, len(passages)):
                    a, b = passages[i], passages[j]
                    adjacency[a][b] += 1.0
                    adjacency[b][a] += 1.0
        result = {src: dict(dst) for src, dst in adjacency.items()}
        _logger.debug("entity graph: %d passages linked over %d entities", len(result), len(entity_to_passages))
        return result
