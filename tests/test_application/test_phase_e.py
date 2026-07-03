"""Tests for Phase E (explanation, LLM-judge loop, result cache, frequency tracker)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


def test_llm_judge_heuristic_score():
    """Without a real LLM, the judge returns a heuristic score in [0, 1]."""
    from src.adapters.services.judge import LLMJudge

    judge = LLMJudge()
    res = judge.score("where is acme", "Acme is located in Berlin", "Acme is located in Berlin")
    assert 0.0 <= res["score"] <= 1.0
    assert res["rationale"]


def test_result_cache_version_invalidation():
    """bump_version evicts entries (corpus-change semantics)."""
    from src.infrastructure.cache.result_cache import ResultCache

    cache = ResultCache(ttl=0)
    payload = object()
    cache.put("q", None, payload)
    assert cache.get("q", None) is payload
    cache.bump_version()
    assert cache.get("q", None) is None


def test_query_frequency_tracker_counts():
    """The tracker counts repeated queries."""
    from src.infrastructure.cache.result_cache import QueryFrequencyTracker

    tracker = QueryFrequencyTracker()
    tracker.track("where acme")
    tracker.track("where acme")
    assert tracker.count("where acme") == 2


@pytest.fixture
def app():
    """Build a throwaway application with Phase E wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_orchestrator_explains_and_caches(app):
    """The orchestrator attaches an explanation and serves the second call from cache."""
    app.ingest_uc.execute(text="Acme Corp is a software company based in Berlin.")
    dto1 = app.query_orchestrator.execute("What is Acme Corp?")
    assert dto1.explanation
    assert not dto1.cached
    dto2 = app.query_orchestrator.execute("What is Acme Corp?")
    assert dto2.cached  # served from cache

    # An ingest invalidates the cache (via the document.ingested event).
    app.ingest_uc.execute(text="Globex is a competitor of Acme.")
    dto3 = app.query_orchestrator.execute("What is Acme Corp?")
    assert not dto3.cached


def test_judge_feedback_updates_fusion(app):
    """judge_and_feedback scores the answer and nudges fusion weights."""
    app.ingest_uc.execute(text="Acme Corp is a software company based in Berlin.")
    dto = app.query_uc.execute("What is Acme Corp?")
    result = app.judge_uc.judge_and_feedback(dto)
    assert 0.0 <= result["score"] <= 1.0
    assert result["feedback_applied"] is True
    assert abs(sum(app._retriever.weights) - 1.0) < 1e-6  # weights stay normalized
