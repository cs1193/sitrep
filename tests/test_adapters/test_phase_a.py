"""Tests for the Phase A retrieval upgrade (PPR, temporal decay, density, wiring)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


def test_ppr_seed_dominates_neighbors():
    """PPR concentrates mass near the seed and decays with graph distance."""
    from src.infrastructure.retrieval.ppr import PPREngine

    adj = {
        "a": {"b": 1},
        "b": {"a": 1, "c": 1},
        "c": {"b": 1, "d": 1},
        "d": {"c": 1},
    }
    scores = PPREngine().run_ppr(adj, ["a"], alpha=0.85)
    # Distribution sums to ~1 and the seed outranks far nodes.
    assert sum(scores.values()) == pytest.approx(1.0, rel=1e-6)
    assert scores["a"] >= scores["c"]
    assert scores["a"] >= scores["d"]


def test_ppr_empty_and_dangling_safe():
    """PPR handles empty graphs and dangling nodes without error."""
    from src.infrastructure.retrieval.ppr import PPREngine

    assert PPREngine().run_ppr({}, []) == {}
    scores = PPREngine().run_ppr({"a": {}, "b": {"a": 1}}, ["b"])
    assert sum(scores.values()) == pytest.approx(1.0, rel=1e-6)


def test_temporal_decay_exponential():
    """Exponential decay is 1 at age 0 and 0.5 at the half-life."""
    from src.infrastructure.retrieval.temporal_scorer import TemporalScorer

    ts = TemporalScorer(strategy="exponential", half_life_days=180)
    assert ts.decay(0) == pytest.approx(1.0)
    assert ts.decay(180) == pytest.approx(0.5)
    assert ts.decay(360) < ts.decay(180)


def test_temporal_blend_weight():
    """The blend applies temporal_weight to the temporal channel."""
    from src.infrastructure.retrieval.temporal_scorer import TemporalScorer

    blended = TemporalScorer.blend({"x": 1.0}, {"x": 0.0}, temporal_weight=0.3)
    assert blended["x"] == pytest.approx(0.3)


@pytest.fixture
def app():
    """Build a throwaway application (Phase A wired via build_application)."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_retriever_exposes_phase_a_signals(app):
    """Retrieved results carry PPR / density / temporal scores in metadata."""
    app.ingest_uc.execute(text="Acme Corp is located in Berlin. Acme Corp makes software.")
    dto = app.query_uc.execute("Where is Acme located?")
    assert dto.results
    meta = dto.results[0].metadata
    assert "ppr_score" in meta
    assert "entity_density" in meta
    assert "temporal_score" in meta
