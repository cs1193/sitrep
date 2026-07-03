"""End-to-end use-case tests (demo mode, core deps only)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


@pytest.fixture
def app():
    """Build a throwaway application against a temp ``.sitrep``."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_ingest_creates_passages_and_facts(app):
    """Ingestion produces passages and at least one fact."""
    result = app.ingest_uc.execute(text="Acme Corp is a software company. Acme Corp is located in Berlin.")
    assert result.passages >= 1
    assert result.facts >= 1
    assert result.schemas == 1


def test_query_returns_answer_and_confidence(app):
    """A query returns an answer and a confidence in [0, 1]."""
    app.ingest_uc.execute(text="Acme Corp is located in Berlin.")
    dto = app.query_uc.execute("Where is Acme located?")
    assert dto.answer
    assert 0.0 <= dto.confidence <= 1.0
    assert dto.full_tokens >= dto.compressed_tokens


def test_positive_feedback_updates_fusion_weights(app):
    """Positive feedback nudges fusion weights while keeping them normalized."""
    app.ingest_uc.execute(text="Acme Corp is located in Berlin.")
    dto = app.query_uc.execute("Where is Acme located?")
    feedback = app.feedback_uc.submit(dto.query_id, "positive", 1.0)
    assert feedback.weights_updated is True
    assert abs(sum(app._retriever.weights) - 1.0) < 1e-6


def test_stats_reports_counts(app):
    """Stats reflects ingested data."""
    app.ingest_uc.execute(text="Widget X is a device. Widget X has a battery.")
    stats = app.stats()
    assert stats.passages >= 1
    assert stats.facts >= 1
