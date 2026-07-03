"""Tests for retrieval fusion, chunking, and extraction adapters."""
from __future__ import annotations

from src.adapters.services.extraction import ExtractionService
from src.infrastructure.chunking import AdaptiveChunker, ChunkingStrategy
from src.infrastructure.retrieval.hybrid_retriever import WeightedFusion, _normalize_scores


def test_fusion_weights_are_normalized():
    """Fusion weights always sum to 1."""
    fusion = WeightedFusion((2, 1, 1))
    assert abs(sum(fusion.weights) - 1.0) < 1e-9


def test_fusion_fuses_scores():
    """Fusion combines per-channel scores linearly."""
    fusion = WeightedFusion((1, 1, 1))
    out = fusion.fuse({"a": 1.0}, {"a": 1.0}, {"a": 1.0})
    assert abs(out["a"] - 1.0) < 1e-6


def test_fusion_online_update_favors_correlated_channel():
    """Positive feedback correlated with the bm25 channel raises its weight."""
    fusion = WeightedFusion((1, 1, 1))
    fusion.update_online([{"x": (1.0, 0.0, 0.0, 1.0)}])
    assert fusion.weights[0] > 1 / 3


def test_normalize_scores_handles_edge_cases():
    """Normalization handles empty, flat, and ranged score dicts."""
    assert _normalize_scores({}) == {}
    ranged = _normalize_scores({"a": 0.0, "b": 1.0})
    assert ranged["a"] == 0.0 and ranged["b"] == 1.0
    flat = _normalize_scores({"a": 5.0, "b": 5.0})
    assert flat["a"] == 1.0 and flat["b"] == 1.0


def test_chunker_fixed_produces_multiple_chunks():
    """Fixed chunking splits a long string into multiple chunks."""
    chunker = AdaptiveChunker(chunk_size=5, overlap=1)
    chunks = chunker.chunk("one two three four five six seven eight", strategy=ChunkingStrategy.FIXED)
    assert len(chunks) >= 2


def test_chunker_empty_returns_empty():
    """Empty input yields no chunks."""
    assert AdaptiveChunker().chunk("", strategy=ChunkingStrategy.FIXED) == []


def test_extraction_regex_finds_facts_and_schema():
    """The regex extractor pulls facts and infers a schema."""
    result = ExtractionService().extract("Acme Corp is a company. The API handles JSON requests.")
    assert len(result.facts) >= 1
    assert result.schema is not None
    assert any(f.subject.lower().startswith("acme") for f in result.facts)
