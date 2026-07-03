#!/usr/bin/env python3
"""SITREP Claude Code integration plugin.

Exposes the SITREP pipeline as a small, importable Python API backed by a
lazily-built :class:`~src.application.Application` singleton. Tools/agents can
call :func:`query`, :func:`ingest`, :func:`train`, :func:`stats`,
:func:`lineage`, and :func:`snapshot` without managing the composition root.

Example::

    import plugin
    plugin.ingest("Acme Corp is located in Berlin.")
    print(plugin.query("Where is Acme?")["answer"])
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_logger = logging.getLogger("sitrep.plugin")
_APP: Any = None


def get_app():
    """Return the lazily-built, cached :class:`Application` singleton."""
    global _APP
    if _APP is None:
        from src.application import build_application

        _APP = build_application()
        _logger.info("SITREP application initialized for plugin")
    return _APP


def reset_app() -> None:
    """Drop the cached application (next call rebuilds it)."""
    global _APP
    if _APP is not None:
        try:
            _APP.close()
        except Exception:  # pragma: no cover
            pass
    _APP = None


# --------------------------------------------------------------------------- API
def query(text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """Answer *text* and return the presenter-formatted result dict."""
    dto = get_app().query_uc.execute(text, top_k=top_k)
    return get_app().query_uc.presenter(dto)


def ingest(text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """Ingest *text* and return the ingestion summary dict."""
    return get_app().ingest_uc.execute(text=text, document_id=document_id).to_dict()


def train(timesteps: Optional[int] = None) -> Dict[str, Any]:
    """Train the compression agent and return the training summary dict."""
    return get_app().train_uc.execute(total_timesteps=timesteps).to_dict()


def stats() -> Dict[str, Any]:
    """Return aggregate system statistics."""
    return get_app().stats().to_dict()


def retrieve(key: str) -> Dict[str, Any]:
    """Headroom CCR ``retrieve`` tool: fetch the uncompressed context for *key*.

    Used for reversible compression — the compressed context is sent to the
    model, while the original is retained locally and can be fetched on demand.
    """
    app = get_app()
    ccr = getattr(app, "ccr_repo", None)
    if ccr is None:
        return {"found": False, "key": key, "error": "CCR repository not configured"}
    entry = ccr.retrieve(key)
    if entry is None:
        return {"found": False, "key": key}
    return {
        "found": True,
        "key": key,
        "content_type": entry.get("content_type"),
        "original": entry.get("original"),
    }


def lineage(decision_id: Optional[str] = None, recent: int = 10) -> Dict[str, Any]:
    """Return a decision trace (if *decision_id*) or recent decisions."""
    app = get_app()
    if decision_id:
        return app.lineage_uc.trace(decision_id).to_dict()
    return {"recent": app.lineage_uc.recent(recent)}


def snapshot(label: Optional[str] = None) -> Dict[str, Any]:
    """Create a data snapshot and return its metadata."""
    from dataclasses import asdict

    return asdict(get_app().version_uc.snapshot(label))


# --------------------------------------------------------------------------- CLI
def main() -> None:
    """Dispatch a single CLI action (e.g. ``plugin.py stats``)."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    action = sys.argv[1]
    if action == "query":
        print(json.dumps(query(" ".join(sys.argv[2:])), indent=2, default=str))
    elif action == "ingest":
        print(json.dumps(ingest(" ".join(sys.argv[2:])), indent=2, default=str))
    elif action == "train":
        print(json.dumps(train(), indent=2, default=str))
    elif action == "stats":
        print(json.dumps(stats(), indent=2, default=str))
    else:
        print(f"Unknown action: {action}\nAvailable: query, ingest, train, stats")


if __name__ == "__main__":  # pragma: no cover
    main()
