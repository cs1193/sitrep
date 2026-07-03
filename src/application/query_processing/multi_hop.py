"""Multi-hop reasoning (Phase D3): answer via graph traversal between entities.

Builds an entity graph from extracted facts, finds a connecting chain between
the query's entities (BFS), gathers passages along the chain, and synthesizes an
answer with an explicit hop explanation.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from src.domain.interfaces import EmbeddingGateway, FactRepository, LLMGateway, Retriever

_logger = logging.getLogger("sitrep.query.multi_hop")

_ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")
# Question/function words to exclude from entity extraction (case-insensitive).
_STOPWORDS = {
    "how", "what", "where", "when", "why", "who", "which", "does", "do", "did",
    "is", "are", "was", "were", "show", "compare", "versus", "and", "of", "to",
    "relate", "relates", "relationship", "image", "picture",
}


class MultiHopReasoner:
    """Answers multi-hop questions by traversing the fact/entity graph."""

    def __init__(
        self,
        fact_repo: FactRepository,
        retriever: Retriever,
        llm: LLMGateway,
        embedder: Optional[EmbeddingGateway] = None,
    ) -> None:
        """Wire the fact repo (graph), retriever (passages), and LLM (synthesis)."""
        self.fact_repo = fact_repo
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder

    def reason(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Return ``{answer, results, chain, entities, confidence}``."""
        entities = self._extract_entities(query)
        facts = self._relevant_facts(entities)
        chain = self._find_chain(entities, facts)
        results = self.retriever.retrieve(query, top_k=top_k)
        passage_ids: set = set()
        for fact in facts:
            passage_ids.update(fact.source_passage_ids)
        context = "\n".join(r.text for r in results)
        chain_text = " -> ".join(chain) if chain else "(no chain found)"
        prompt = (
            f"Context:\n{context}\n\n"
            f"Entity chain: {chain_text}\n\n"
            f"Question: {query}\nAnswer using the chain of facts above:"
        )
        try:
            answer = self.llm.generate(prompt)
        except Exception as exc:  # pragma: no cover
            _logger.warning("multi-hop answer generation failed: %s", exc)
            answer = f"Connected via: {chain_text}"
        confidence = round(min(1.0, 0.4 + 0.15 * max(0, len(chain) - 1)), 4)
        return {
            "answer": answer,
            "results": results,
            "chain": chain,
            "entities": entities,
            "confidence": confidence,
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _extract_entities(query: str) -> List[str]:
        """Return candidate entities (capitalized tokens, then content words)."""
        caps = [
            m.group(1)
            for m in _ENTITY_RE.finditer(query or "")
            if m.group(1).lower() not in _STOPWORDS
        ]
        if caps:
            # dedupe preserving order
            seen, out = set(), []
            for c in caps:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    out.append(c)
            return out
        # fallback: significant content words
        return [t for t in (query or "").split() if len(t) > 4][:3]

    def _relevant_facts(self, entities: List[str]) -> List[Any]:
        """Return facts mentioning any of *entities* (or all valid if none)."""
        if not entities:
            try:
                return self.fact_repo.all_valid()
            except Exception:  # pragma: no cover
                return []
        facts = []
        seen = set()
        for entity in entities:
            for fact in self.fact_repo.search(entity):
                if fact.id not in seen:
                    seen.add(fact.id)
                    facts.append(fact)
        return facts

    @staticmethod
    def _entity_graph(facts) -> Dict[str, set]:
        """Build an undirected entity graph {entity: {neighbor entities}} from facts."""
        graph: Dict[str, set] = defaultdict(set)
        for fact in facts:
            s = fact.subject.lower()
            o = fact.object_value.lower()
            if s and o and s != o:
                graph[s].add(o)
                graph[o].add(s)
        return graph

    def _find_chain(self, entities: List[str], facts) -> List[str]:
        """BFS shortest entity-path between the first two query entities."""
        if len(entities) < 2:
            return list(entities)
        start, target = entities[0].lower(), entities[1].lower()
        graph = self._entity_graph(facts)
        if start not in graph or target not in graph:
            return [entities[0], entities[1]]
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for nbr in graph.get(node, ()):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append((nbr, path + [nbr]))
        return [entities[0], entities[1]]
