"""Tests for the Headroom compression layer (compressors, router, CCR, pipeline)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.utils.config import SitrepConfig, set_config


# --------------------------------------------------------------------------- unit: compressors
def test_smart_crusher_strips_redundant_and_shrinks():
    """SmartCrusher drops null/empty values and reduces JSON size."""
    from src.infrastructure.compression import SmartCrusher

    data = json.dumps({"a": {"b": 1, "c": None, "d": ""}, "items": list(range(20))})
    out = SmartCrusher(max_array=5).compress(data, ratio=0.5)
    assert len(out) < len(data)
    assert "null" not in out  # None values are dropped


def test_code_compressor_strips_comments_and_docstrings():
    """CodeCompressor removes comments/docstrings while keeping structure."""
    from src.infrastructure.compression import CodeCompressor

    code = '"""module doc"""\n# a comment\ndef f():\n    """inner"""\n    return 1  # r\n'
    out = CodeCompressor().compress(code, ratio=0.4)
    assert "#" not in out
    assert "module doc" not in out
    assert "def f" in out


def test_log_compressor_strips_timestamps_and_traceback():
    """LogCompressor removes timestamps, tracebacks, and dedupes."""
    from src.infrastructure.compression import LogCompressor

    log = (
        "2024-01-01T10:00:00Z INFO start\n"
        "2024-01-01T10:00:01Z INFO start\n"
        "Traceback (most recent call last):\n"
        "  File x\n"
        "2024-01-01T10:00:02Z INFO end\n"
    )
    out = LogCompressor().compress(log, ratio=0.8)
    assert "2024-01-01" not in out
    assert "Traceback" not in out
    assert "×2" in out  # deduplicated consecutive lines


# --------------------------------------------------------------------------- unit: router
def test_router_detects_each_content_type():
    """ContentRouter classifies JSON, code, logs, and text correctly."""
    from src.infrastructure.compression import ContentRouter, ContentType

    router = ContentRouter()
    assert router.detect(json.dumps({"a": 1})) == ContentType.JSON
    assert router.detect("def f():\n    return 1\n") == ContentType.CODE
    assert (
        router.detect(
            "2024-01-01T10:00:00Z INFO a\n2024-01-01T10:00:01Z WARN b\n2024-01-01T10:00:02Z ERROR c\n"
        )
        == ContentType.LOG
    )
    assert router.detect("Acme is a company based in Berlin.") == ContentType.TEXT


def test_router_compress_returns_metadata():
    """ContentRouter.compress returns telemetry with type and token counts."""
    from src.infrastructure.compression import ContentRouter

    router = ContentRouter()
    out, meta = router.compress(json.dumps({"a": 1, "b": [1, 2, 3]}), ratio=0.5)
    assert meta["content_type"] == "json"
    assert meta["compressor"] == "smart_crusher"
    assert meta["original_tokens"] >= meta["compressed_tokens"]


# --------------------------------------------------------------------------- unit: cache aligner
def test_cache_aligner_prefix_is_stable():
    """CacheAligner yields a byte-identical prefix across different user content."""
    from src.infrastructure.cache import CacheAligner

    aligner = CacheAligner()
    sys_a, ua, meta_a = aligner.align("user content A")
    sys_b, ub, meta_b = aligner.align("user content B")
    assert sys_a == sys_b
    assert meta_a["prefix_hash"] == meta_b["prefix_hash"]
    assert meta_a["cache_eligible"] is True
    assert ua != ub


# --------------------------------------------------------------------------- unit: CCR repo
def test_ccr_store_retrieve_and_expiry(tmp_path):
    """CCR repository stores, retrieves, and expires entries by TTL."""
    from src.adapters.repositories.ccr_repo import SQLiteCCRRepository
    from src.infrastructure.db.sqlite_client import SQLiteClient

    cfg = SitrepConfig(base_dir=str(tmp_path / ".sitrep"), _env_file=None)
    set_config(cfg)
    cfg.ensure_directories()
    client = SQLiteClient.from_config()
    try:
        ccr = SQLiteCCRRepository(client, default_ttl=3600)
        key = ccr.store("original content", "compressed", content_type="text")
        assert ccr.retrieve(key)["original"] == "original content"
        expired = ccr.store("ephemeral", "c", ttl=0)
        assert ccr.retrieve(expired) is None
    finally:
        client.close()


# --------------------------------------------------------------------------- integration: pipeline
@pytest.fixture
def app():
    """Build a throwaway application against a temp ``.sitrep``."""
    cfg = SitrepConfig(base_dir=os.path.join(tempfile.mkdtemp(), ".sitrep"), _env_file=None)
    set_config(cfg)
    from src.application import build_application

    application = build_application(config=cfg)
    yield application
    application.close()


def test_query_pipeline_populates_headroom_telemetry(app):
    """The query use case routes content, stores CCR, and reports cache eligibility."""
    app.ingest_uc.execute(text="Acme Corp is a software company. Acme Corp is located in Berlin.")
    dto = app.query_uc.execute("Where is Acme located?")
    assert dto.content_type in ("text", "json", "code", "log")
    assert dto.compressor
    assert dto.ccr_key is not None
    assert dto.cache_eligible is True
    entry = app.ccr_repo.retrieve(dto.ccr_key)
    assert entry is not None
    assert "Berlin" in entry["original"]
