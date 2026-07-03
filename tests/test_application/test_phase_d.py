"""Tests for Phase D query intelligence (intent, decomposition, multi-hop, orchestrator)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


def test_intent_classifier_routes_by_cues():
    """Keyword cues map queries to the expected intent."""
    from src.application.query_processing.intent import IntentClassifier, IntentType

    clf = IntentClassifier()
    assert clf.classify("compare Acme vs Globex") == IntentType.COMPARISON
    assert clf.classify("show me an image of the widget") == IntentType.MULTIMODAL
    assert clf.classify("how does Acme relate to Globex?") == IntentType.MULTI_HOP
    assert clf.classify("a simple factual lookup query") == IntentType.SIMPLE


def test_query_decomposer_splits_compound():
    """Conjunctions split a compound query into sub-queries."""
    from src.application.query_processing.decomposition import QueryDecomposer

    dec = QueryDecomposer()
    parts = dec.decompose("What is Acme? and Where is Globex located?")
    assert len(parts) == 2
    assert any("acme" in p.lower() for p in parts)
    # A simple query is returned as a single-element list.
    assert dec.decompose("just one question") == ["just one question"]


@pytest.fixture
def app():
    """Build a throwaway application with Phase D wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_orchestrator_simple_passthrough(app):
    """A simple query is routed through QueryKnowledgeUseCase with intent=simple."""
    app.ingest_uc.execute(text="Acme Corp is a software company based in Berlin.")
    dto = app.query_orchestrator.execute("What is Acme Corp?")
    assert dto.intent == "simple"
    assert dto.answer


def test_orchestrator_multi_hop_builds_chain(app):
    """A multi-hop query traverses the entity graph and reports a hop chain."""
    app.ingest_uc.execute(text="Acme has Berlin.")
    app.ingest_uc.execute(text="Berlin has Germany.")
    dto = app.query_orchestrator.execute("How does Acme relate to Germany?")
    assert dto.intent == "multi_hop"
    chain = dto.extras.get("hop_chain", [])
    # The chain connects acme -> berlin -> germany.
    assert "berlin" in [c.lower() for c in chain]
