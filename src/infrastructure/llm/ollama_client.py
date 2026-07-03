"""Ollama LLM client over HTTP (stdlib ``urllib``; no SDK dependency).

Talks to a local Ollama server (``/api/generate``). Works whether or not the
``ollama`` Python package is installed.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.llm.ollama")


class OllamaLLMClient(LLMGateway):
    """Local LLM via the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        """Configure endpoint, model, and request timeout."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = "ollama"

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to ``/api/tags``."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=min(self.timeout, 10.0)) as resp:
                return resp.status == 200
        except Exception as exc:
            _logger.debug("ollama not reachable at %s: %s", self.base_url, exc)
            return False

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Call ``/api/generate`` and return the model's response text."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.2)),
                "num_predict": int(kwargs.get("max_new_tokens", 256)),
            },
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return (body.get("response") or "").strip()
        except urllib.error.URLError as exc:
            _logger.error("ollama generate failed: %s", exc)
            raise RuntimeError(f"ollama request failed: {exc}") from exc
