#!/usr/bin/env python3
"""Launch the SITREP Gradio web UI (extra ``[web]``)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Build the application and launch the web UI."""
    from src.application import build_application
    from src.presentation.web import launch

    app = build_application()
    print("Starting SITREP web UI at http://127.0.0.1:7860 ...")
    launch(application=app, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
