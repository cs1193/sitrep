"""Tests for Phase B memory hygiene (entity graph, importance, consolidation, forgetting)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


def test_entity_graph_links_shared_entities():
    """Passages sharing a fact entity get a bidirectional edge."""
    from src.domain.schemas import Fact
    from src.infrastructure.retrieval.entity_graph import EntityGraphBuilder

    f1 = Fact(subject="acme", predicate="located", object_value="berlin", source_passage_ids=["p1", "p2"])
    f2 = Fact(subject="acme", predicate="makes", object_value="software", source_passage_ids=["p2", "p3"])
    adj = EntityGraphBuilder.build([f1, f2])
    # entity "acme" connects p1, p2, p3 → all pairs linked.
    assert "p2" in adj.get("p1", {})
    assert "p3" in adj.get("p2", {})
    assert "p3" in adj.get("p1", {})
    assert adj["p1"]["p2"] > 0  # symmetric weight


def test_importance_scorer_stashes_in_metadata():
    """ImportanceScorer returns a score in [0,1] and writes it to metadata."""
    from src.adapters.services.importance import ImportanceScorer
    from src.domain.schemas import Passage

    scorer = ImportanceScorer()
    passage = Passage(text="hello world", document_id="d")
    imp = scorer.score_passage(passage, age_days=0, max_access_count=5)
    assert 0.0 <= imp <= 1.0
    assert passage.metadata["importance"] == imp


def test_forgetting_criteria_reason_and_strategy():
    """Criteria classify correctly and map to non-destructive strategies."""
    from src.domain.forgetting import ForgettingCriteria, ForgettingReason, ForgettingStrategy

    criteria = ForgettingCriteria()
    assert criteria.strategy_for(ForgettingReason.LOW_IMPORTANCE) == ForgettingStrategy.GRADUAL_FADING
    assert criteria.strategy_for(ForgettingReason.OBSOLESCENCE) == ForgettingStrategy.SOFT_DELETE
    assert criteria.reason_for(
        age_days=400, inactive_days=0, importance=0.9, access_frequency=1, redundancy_count=0
    ) == ForgettingReason.OBSOLESCENCE
    assert criteria.reason_for(
        age_days=0, inactive_days=0, importance=0.9, access_frequency=1, redundancy_count=0
    ) == ForgettingReason.NONE
    assert criteria.strategy_for(ForgettingReason.NONE) == ForgettingStrategy.KEEP


@pytest.fixture
def app():
    """Build a throwaway application with Phase B wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_consolidation_merges_near_duplicates(app):
    """Two near-identical passages collapse: the loser is SOFT_DELETED."""
    from src.domain.schemas import Passage

    emb = app._embedder.embed("Acme Corp makes software in Berlin.")
    p1 = Passage(text="Acme Corp makes software in Berlin.", document_id="d", id="p1", embedding=emb)
    p2 = Passage(text="Acme Corp makes software in Berlin.", document_id="d", id="p2", embedding=emb)
    app.passage_repo.save(p1)
    app.passage_repo.save(p2)
    report = app.consolidation_uc.execute(limit=10)
    assert report["merged"] >= 1
    statuses = [
        app.passage_repo.get(pid).metadata.get("memory_status", "active") for pid in ("p1", "p2")
    ]
    assert "soft_deleted" in statuses


def test_forgetting_dry_run_then_apply(app):
    """Dry-run classifies without mutating; apply changes status but never hard-deletes."""
    from src.domain.schemas import Passage

    passage = Passage(
        text="some old content here",
        document_id="d",
        id="p3",
        embedding=app._embedder.embed("some old content here"),
    )
    passage.metadata["importance"] = 0.05  # below min_importance (0.2) → LOW_IMPORTANCE
    app.passage_repo.save(passage)

    dry = app.forgetting_uc.execute(dry_run=True)
    assert any(a["strategy"] != "keep" for a in dry["actions"])
    assert app.passage_repo.get("p3").metadata.get("memory_status", "active") == "active"

    app.forgetting_uc.execute(dry_run=False)
    status = app.passage_repo.get("p3").metadata.get("memory_status")
    assert status in ("fading", "soft_deleted", "archived")
    assert status != "permanently_deleted"  # never a hard delete by default
