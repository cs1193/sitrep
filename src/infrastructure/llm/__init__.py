"""LLM gateway factory with an Ollama → Transformers → DEMO cascade.

The ``DEMO`` backend requires no model and produces deterministic, context-aware
templated output so the system is fully functional without any download.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.domain.interfaces import LLMGateway
from src.utils.config import SitrepConfig

_logger = logging.getLogger("sitrep.llm")


class DemoLLMClient(LLMGateway):
    """Deterministic, dependency-free LLM stand-in.

    Produces extractive answers from any "Context:" section in the prompt, which
    lets the full query/answer pipeline work without a real model.
    """

    name = "demo"

    def is_available(self) -> bool:
        """Always available."""
        return True

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Return a deterministic answer derived from context embedded in *prompt*."""
        context = ""
        m = re.search(r"[Cc]ontext:?\s*(.*?)(?:\n[A-Z][a-z]+:|$)", prompt, re.DOTALL)
        if m:
            context = m.group(1).strip()
        question = ""
        q = re.search(r"[Qq]uestion:?\s*(.+)", prompt)
        if q:
            question = q.group(1).strip()

        if context:
            first_sentence = re.split(r"(?<=[.!?])\s+", context.strip())[0][:240]
            return f"Based on the available context: {first_sentence}"
        if question:
            return f"[demo] I would answer using retrieved evidence for: {question[:160]}"
        return "[demo] no actionable input provided"


def get_llm_gateway(config: Optional[SitrepConfig] = None) -> LLMGateway:
    """Resolve the active LLM gateway according to ``cfg.llm_provider``.

    Cascade order for ``auto``: Ollama → Transformers → Demo.
    """
    from src.utils.config import get_config

    cfg = config or get_config()
    provider = cfg.llm_provider

    if provider == "demo":
        return DemoLLMClient()
    if provider == "ollama":
        client = _make_ollama(cfg)
        if client is not None:
            return client
        _logger.warning("Ollama unavailable; falling back to auto cascade")
        return _auto(cfg)
    if provider == "transformers":
        client = _make_transformers(cfg)
        if client is not None:
            return client
        _logger.warning("Transformers unavailable; falling back to auto cascade")
        return _auto(cfg)
    return _auto(cfg)


def _make_ollama(cfg: SitrepConfig) -> Optional[LLMGateway]:
    """Construct an Ollama client if the server is reachable."""
    try:
        from src.infrastructure.llm.ollama_client import OllamaLLMClient

        client = OllamaLLMClient(cfg.ollama_url, cfg.ollama_model, cfg.ollama_timeout)
        return client if client.is_available() else None
    except Exception as exc:  # pragma: no cover
        _logger.debug("ollama construction failed: %s", exc)
        return None


def _make_transformers(cfg: SitrepConfig) -> Optional[LLMGateway]:
    """Construct a Transformers client if importable."""
    try:
        from src.infrastructure.llm.transformers_client import TransformersLLMClient

        client = TransformersLLMClient(
            cfg.hf_llm_model, cfg.llm_max_new_tokens, cfg.llm_temperature
        )
        return client if client.is_available() else None
    except Exception as exc:  # pragma: no cover
        _logger.debug("transformers construction failed: %s", exc)
        return None


def _auto(cfg: SitrepConfig) -> LLMGateway:
    """Resolve the best available backend (Ollama → Transformers → Demo)."""
    client = _make_ollama(cfg)
    if client is not None:
        _logger.info("LLM gateway: ollama (%s)", cfg.ollama_model)
        return client
    client = _make_transformers(cfg)
    if client is not None:
        _logger.info("LLM gateway: transformers (%s)", cfg.hf_llm_model)
        return client
    _logger.info("LLM gateway: demo (no model required)")
    return DemoLLMClient()


__all__ = ["DemoLLMClient", "get_llm_gateway"]
