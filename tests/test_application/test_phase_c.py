"""Tests for Phase C write safety (Unit of Work, event log, provenance edges)."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


def _boom():
    raise RuntimeError("simulated kuzu failure")


@pytest.fixture
def app():
    """Build a throwaway application with Phase C wired."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_unit_of_work_commits_on_success(app):
    """A clean UoW commits the SQLite write."""
    client = app.client
    with app.unit_of_work() as uow:
        client.execute(
            "INSERT INTO skills (id, name, created_at) VALUES (?,?,?)",
            ("uow_ok", "x", "2026-01-01T00:00:00+00:00"),
        )
        uow.register(do=lambda: None)  # non-SQLite op succeeds
    row = client.fetchone("SELECT id FROM skills WHERE id=?", ("uow_ok",))
    assert row is not None


def test_unit_of_work_rolls_back_on_non_sqlite_failure(app):
    """A failing non-SQLite op rolls back the SQLite write (+ compensates)."""
    client = app.client
    undone = {"flag": False}

    def _undo():
        undone["flag"] = True

    with pytest.raises(RuntimeError):
        with app.unit_of_work() as uow:
            client.execute(
                "INSERT INTO skills (id, name, created_at) VALUES (?,?,?)",
                ("uow_rollback", "x", "2026-01-01T00:00:00+00:00"),
            )
            uow.register(do=lambda: None, undo=_undo)  # succeeds, then compensated
            uow.register(do=_boom)  # fails → rollback + undo prior
    # SQLite write rolled back.
    assert client.fetchone("SELECT id FROM skills WHERE id=?", ("uow_rollback",)) is None
    # Compensation ran for the prior successful op.
    assert undone["flag"] is True


def test_event_log_append_and_replay(app, tmp_path):
    """EventLog appends JSONL and replays events in order."""
    from src.infrastructure.persistence import EventLog

    log = EventLog(tmp_path / "events.jsonl")
    log.append({"topic": "t", "payload": {"a": 1}})
    log.append({"topic": "t", "payload": {"a": 2}})
    seen = []
    n = log.replay(lambda e: seen.append(e["payload"]["a"]))
    assert n == 2
    assert seen == [1, 2]


def test_provenance_recorded_on_conflict(app):
    """Conflicting facts record a SUPERSEDES edge (attributes + lineage decision)."""
    app.ingest_uc.execute(text="Acme Corp is a bank.")
    app.ingest_uc.execute(text="Acme Corp is a school.")

    decisions = app.lineage_uc.recent(50)
    supersede = [d for d in decisions if d.get("action") == "supersede"]
    assert supersede, "expected a supersede decision after conflicting ingests"

    invalidated_id = supersede[0]["outputs"]["invalidated"]
    kept_id = supersede[0]["outputs"]["kept"]
    loser = app.fact_repo.get(invalidated_id)
    assert loser is not None
    assert loser.attributes.get("superseded_by") == kept_id


def test_extraction_handles_numeric_facts():
    """Numeric-leading objects (prices/quantities/years) extract as facts."""
    from src.adapters.services.extraction import ExtractionService

    result = ExtractionService().extract("Acme revenue is 100 dollars.")
    objects = [f.object_value for f in result.facts]
    assert any("100" in o for o in objects), f"numeric fact not extracted: {objects}"
