"""Adaptive compression service.

Reduces context to a target token budget via extractive selection (sentence
scoring by query similarity / centrality), with an optional LLM summarization
path for aggressive ratios. Reports token savings for metrics.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple

from src.domain.interfaces import EmbeddingGateway, LLMGateway
from src.utils.common import cosine_similarity, count_tokens_heuristic, normalize

_logger = logging.getLogger("sitrep.services.compression")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class CompressionService:
    """Token-budgeted context compression."""

    def __init__(
        self,
        embedder: Optional[EmbeddingGateway] = None,
        llm: Optional[LLMGateway] = None,
    ) -> None:
        """Store optional embedder (for scoring) and LLM (for summarization)."""
        self.embedder = embedder
        self.llm = llm

    def compress(
        self,
        text: str,
        ratio: float = 0.5,
        query: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """Compress *text* to ~*ratio* of its tokens; return (text, full, compressed)."""
        ratio = max(0.05, min(1.0, float(ratio)))
        full_tokens = count_tokens_heuristic(text)
        if full_tokens == 0:
            return ("", 0, 0)
        if ratio >= 0.999:
            return (text, full_tokens, full_tokens)

        target = max(1, int(full_tokens * ratio))
        # Try LLM summarization for aggressive compression when available.
        if ratio <= 0.4 and self._can_summarize():
            try:
                summary = self._summarize(text, target, query)
                if summary:
                    comp_tokens = count_tokens_heuristic(summary)
                    return (summary, full_tokens, comp_tokens)
            except Exception as exc:  # pragma: no cover
                _logger.warning("LLM summarization failed, using extractive: %s", exc)

        compressed = self._extractive(text, target, query)
        return (compressed, full_tokens, count_tokens_heuristic(compressed))

    def compress_results(
        self,
        texts: Sequence[str],
        ratio: float = 0.5,
        query: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """Compress a sequence of retrieved passages as one combined context."""
        combined = "\n\n".join(t for t in texts if t)
        return self.compress(combined, ratio, query)

    # ----------------------------------------------------------------- internals
    def _extractive(self, text: str, target_tokens: int, query: Optional[str]) -> str:
        """Select the highest-scoring sentences up to *target_tokens*, in order."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return text
        if len(sentences) == 1:
            return sentences[0]

        embeddings = self._embed_sentences(sentences)
        query_vec = self._embed_query(query) if query else self._centroid(embeddings)
        scores = [
            (i, cosine_similarity(query_vec, emb)) for i, emb in enumerate(embeddings)
        ]
        # Tie-break: longer (more informative) sentences first.
        scores.sort(
            key=lambda x: (x[1], len(sentences[x[0]])),
            reverse=True,
        )

        chosen = set()
        used = 0
        for idx, _score in scores:
            tok = count_tokens_heuristic(sentences[idx])
            if used + tok > target_tokens and chosen:
                continue
            chosen.add(idx)
            used += tok
            if used >= target_tokens:
                break
        return " ".join(sentences[i] for i in sorted(chosen))

    def _embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences via the configured embedder or a hash fallback."""
        if self.embedder is not None:
            return self.embedder.embed_batch(sentences)
        from src.utils.embedding import hash_embedding

        return [hash_embedding(s) for s in sentences]

    def _embed_query(self, query: str) -> List[float]:
        """Embed the query (normalized) for sentence scoring."""
        if self.embedder is not None:
            return normalize(self.embedder.embed(query))
        from src.utils.embedding import hash_embedding

        return normalize(hash_embedding(query))

    @staticmethod
    def _centroid(embeddings: List[List[float]]) -> List[float]:
        """Return the normalized mean of *embeddings* (centrality scoring)."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        acc = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                acc[i] += float(v)
        n = float(len(embeddings))
        return normalize([v / n for v in acc])

    def _can_summarize(self) -> bool:
        """Return True if a non-demo LLM is available for summarization."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _summarize(self, text: str, target_tokens: int, query: Optional[str]) -> Optional[str]:
        """Ask the LLM to summarize *text* to roughly *target_tokens*."""
        focus = f" Focus on: {query}." if query else ""
        prompt = (
            f"Summarize the following context in at most {target_tokens} tokens.{focus}\n"
            f"Context:\n{text[:3000]}"
        )
        return self.llm.generate(prompt) or None
