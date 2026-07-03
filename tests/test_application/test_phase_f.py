"""Tests for Phase F (Allen interval algebra + anomaly detection)."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.utils.config import SitrepConfig, set_config


# ----------------------------------------------------------------- Allen (F4)
def test_allen_basic_relations():
    """The 13 Allen relations classify correctly on constructed intervals."""
    from src.infrastructure.reasoning.temporal_allen import AllenRelation, allen_relation, inverse

    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2020, 6, 1, tzinfo=timezone.utc)
    t2 = datetime(2021, 1, 1, tzinfo=timezone.utc)
    t3 = datetime(2021, 6, 1, tzinfo=timezone.utc)

    assert allen_relation(t0, t1, t2, t3) == AllenRelation.BEFORE           # [0,1] before [2,3]
    assert allen_relation(t2, t3, t0, t1) == AllenRelation.AFTER            # inverse
    assert allen_relation(t0, t2, t1, t3) == AllenRelation.OVERLAPS         # [0,2] overlaps [1,3]
    assert allen_relation(t1, t2, t0, t3) == AllenRelation.DURING           # [1,2] during [0,3]
    assert allen_relation(t0, t3, t1, t2) == AllenRelation.CONTAINS         # [0,3] contains [1,2]
    assert allen_relation(t0, t1, t0, t2) == AllenRelation.STARTS           # same start, A ends first
    assert allen_relation(t0, t2, t0, t1) == AllenRelation.STARTED_BY
    assert allen_relation(t1, t3, t0, t3) == AllenRelation.FINISHES         # same end, A starts later
    assert allen_relation(t0, t3, t0, t3) == AllenRelation.EQUALS
    assert inverse(AllenRelation.BEFORE) == AllenRelation.AFTER
    assert inverse(AllenRelation.DURING) == AllenRelation.CONTAINS


@pytest.fixture
def app():
    """Build a throwaway application with Phase F wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_temporal_reasoning_relate(app):
    """Two facts with explicit validity windows relate via Allen."""
    from src.domain.schemas import Fact

    a = Fact(subject="a", predicate="p", object_value="b",
             valid_from="2020-01-01T00:00:00+00:00", valid_to="2020-12-31T00:00:00+00:00")
    b = Fact(subject="c", predicate="p", object_value="d",
             valid_from="2021-06-01T00:00:00+00:00", valid_to="2022-01-01T00:00:00+00:00")
    app.fact_repo.save(a)
    app.fact_repo.save(b)
    result = app.temporal_reasoning_uc.relate(a.id, b.id)
    assert result["relation"] == "before"
    assert result["inverse"] == "after"


# ----------------------------------------------------------------- Anomaly (F3)
def test_anomaly_detector_flags_outlier():
    """A clear outlier in a 20+1 sample is flagged above threshold."""
    from src.adapters.services.anomaly import AnomalyDetector
    from src.domain.schemas import Passage

    passages = [
        Passage(text=f"normal passage number {i}", document_id="d", id=f"n{i}",
                metadata={"access_count": 0, "importance": 0.5})
        for i in range(20)
    ]
    passages.append(Passage(text="weird outlier passage", document_id="d", id="outlier",
                            metadata={"access_count": 1000, "importance": 0.5}))
    detector = AnomalyDetector(threshold=2.0)
    anomalies = detector.detect(passages)
    flagged_ids = {a["passage_id"] for a in anomalies}
    assert "outlier" in flagged_ids
    assert any(a["signal"] == "access_count" and a["passage_id"] == "outlier" for a in anomalies)


def test_anomaly_usecase_runs(app):
    """The anomaly use case scans passages and returns structured output."""
    app.ingest_uc.execute(text="Acme Corp is a software company based in Berlin.")
    report = app.anomaly_uc.execute()
    assert "n_scanned" in report
    assert isinstance(report["anomalies"], list)
    assert isinstance(report["by_signal"], dict)
