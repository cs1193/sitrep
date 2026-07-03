"""Tests for Phase G (causal inference + multimodal images/audio)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


# ----------------------------------------------------------------- causal (G3)
def test_causal_effect_direct_and_mediated():
    """Direct effect + mediated effect sum correctly over directed paths."""
    from src.domain.causal import CausalGraph
    from src.infrastructure.reasoning.causal import CausalEngine

    g = CausalGraph()
    g.add_edge("x", "y", weight=0.7)        # direct
    g.add_edge("x", "m", weight=0.5)
    g.add_edge("m", "y", weight=0.6)        # mediated x->m->y
    eng = CausalEngine(g)
    # total effect of x on y = 0.7 + (0.5*0.6) = 1.0
    assert eng.causal_effect("x", "y") == pytest.approx(1.0)


def test_causal_confounders_detected():
    """A common cause of treatment and outcome is a backdoor confounder."""
    from src.domain.causal import CausalGraph
    from src.infrastructure.reasoning.causal import CausalEngine

    g = CausalGraph()
    g.add_edge("z", "x", weight=0.5)        # z causes treatment
    g.add_edge("z", "y", weight=0.3)        # z causes outcome (confounder)
    g.add_edge("x", "y", weight=0.4)
    eng = CausalEngine(g)
    # causal effect is the directed x->y only (z is upstream of x, not a path from x).
    assert eng.causal_effect("x", "y") == pytest.approx(0.4)
    assert "z" in eng.confounders("x", "y")


def test_counterfactual_linear_estimate():
    """Counterfactual outcome shifts by effect * (x' - x)."""
    from src.domain.causal import CausalGraph
    from src.infrastructure.reasoning.causal import CausalEngine

    g = CausalGraph()
    g.add_edge("treatment", "outcome", weight=0.5)
    eng = CausalEngine(g)
    cf = eng.counterfactual("treatment", "outcome", factual_value=2.0,
                            intervention_value=4.0, factual_outcome=10.0)
    # delta = 0.5 * (4 - 2) = 1.0 → estimated_outcome = 11.0
    assert cf["delta"] == pytest.approx(1.0)
    assert cf["estimated_outcome"] == pytest.approx(11.0)
    assert "explanation" in cf and cf["confidence"] > 0


# ----------------------------------------------------------------- multimodal (G1/G2)
@pytest.fixture
def app():
    """Build a throwaway application with Phase G wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_clip_fallback_embeds_caption(app):
    """Without CLIP, embed_text/embed_image fall back to caption hashing (same space)."""
    txt = app.clip_embedder.embed_text("a cat on a mat")
    img = app.clip_embedder.embed_image("a cat on a mat")
    assert len(txt) == len(img) and len(txt) > 0


def test_cross_modal_retrieval_links_image_to_query(app):
    """An ingested image is retrievable by a caption-like query via cross-modal similarity."""
    app.multimodal_uc.ingest_image("a cat sitting on a sofa", linked_passage_ids=["p1"])
    app.multimodal_uc.ingest_image("a car on a highway")
    result = app.multimodal_uc.retrieve_cross_modal("cat sofa")
    assert result["images"]
    top = result["images"][0]
    assert "cat" in top["caption"]
    assert top["score"] > 0
    assert "p1" in top["linked_passages"]


def test_audio_ingest_stores_media(app):
    """An audio asset is embedded via its transcript and stored."""
    from src.domain.multimodal.av_entities import TemporalSegment

    seg = TemporalSegment(start=0.0, end=2.5, text="hello world")
    out = app.multimodal_uc.ingest_audio("hello world transcript", segments=[seg])
    assert out["audio_id"]
    assert app.mm_repo.count_media(kind="audio") >= 1
