"""Adaptive chunker: semantic, fixed-size, and paragraph strategies.

Semantic chunking detects topic boundaries via embedding similarity between
consecutive sentences. Falls back gracefully to fixed-size or paragraph
chunking when no embedder is supplied.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import List, Optional

from src.domain.interfaces import EmbeddingGateway
from src.utils.common import cosine_similarity, count_tokens_heuristic

_logger = logging.getLogger("sitrep.chunking")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    AUTO = "auto"
    SEMANTIC = "semantic"
    FIXED = "fixed"
    PARAGRAPH = "paragraph"


class AdaptiveChunker:
    """Multi-strategy chunker producing token-budgeted text chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_size: int = 32,
        embedder: Optional[EmbeddingGateway] = None,
        semantic_threshold: float = 0.40,
    ) -> None:
        """Configure token budgets, optional embedder, and semantic boundary threshold."""
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold

    # ----------------------------------------------------------------- primitives
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split *text* into sentences on terminal punctuation or newlines."""
        return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]

    def fixed(self, text: str) -> List[str]:
        """Word-level fixed-size chunking with overlap."""
        words = (text or "").split()
        if not words:
            return []
        chunks: List[str] = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + self.chunk_size]))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def paragraph(self, text: str) -> List[str]:
        """Chunk by blank-line separated paragraphs."""
        paras = [p.strip() for p in _PARAGRAPH_SPLIT.split(text or "") if p.strip()]
        # Merge tiny paragraphs to respect min_size.
        merged: List[str] = []
        for p in paras:
            if merged and count_tokens_heuristic(merged[-1]) < self.min_size:
                merged[-1] = merged[-1] + "\n" + p
            else:
                merged.append(p)
        return merged

    def semantic(self, text: str, threshold: Optional[float] = None) -> List[str]:
        """Chunk by detecting semantic boundaries via embedding similarity."""
        sentences = self.split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text and text.strip() else []
        threshold = self.semantic_threshold if threshold is None else threshold
        if self.embedder is not None:
            embeddings = self.embedder.embed_batch(sentences)
        else:
            from src.utils.embedding import hash_embedding

            embeddings = [hash_embedding(s) for s in sentences]

        chunks: List[str] = []
        current = [sentences[0]]
        cur_tokens = count_tokens_heuristic(sentences[0])
        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            tok = count_tokens_heuristic(sentences[i])
            boundary = sim < threshold or (cur_tokens + tok) > self.chunk_size
            if boundary:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                cur_tokens = tok
            else:
                current.append(sentences[i])
                cur_tokens += tok
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if count_tokens_heuristic(c) >= 1]

    # ----------------------------------------------------------------- dispatch
    def chunk(self, text: str, strategy: ChunkingStrategy = ChunkingStrategy.AUTO) -> List[str]:
        """Chunk *text* using *strategy* (``auto`` picks the best available)."""
        if not text or not text.strip():
            return []
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)
        if strategy == ChunkingStrategy.SEMANTIC:
            return self.semantic(text)
        if strategy == ChunkingStrategy.PARAGRAPH:
            return self.paragraph(text)
        if strategy == ChunkingStrategy.FIXED:
            return self.fixed(text)
        # AUTO: prefer paragraphs when the text is multi-paragraph, else semantic.
        if len(self.paragraph(text)) > 1:
            return self.paragraph(text)
        if self.embedder is not None:
            return self.semantic(text)
        return self.fixed(text)
