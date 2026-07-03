"""Precompute transformer KV caches for passages (extra ``[llm]`` + torch).

Runs a single forward pass per passage to capture ``past_key_values`` and
stores them (pickled BLOB) via the KV-cache repository. Requires
``transformers`` and ``torch``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from src.domain.interfaces import KVCacheRepository, PassageRepository

_logger = logging.getLogger("sitrep.kv.precomputer")


class KVCachePrecomputer:
    """Precompute and persist KV caches for passages."""

    def __init__(
        self,
        model_name: str,
        kv_repo: KVCacheRepository,
        passage_repo: Optional[PassageRepository] = None,
        device: str = "cpu",
    ) -> None:
        """Store config; the model loads lazily on first :meth:`precompute`."""
        self.model_name = model_name
        self.kv_repo = kv_repo
        self.passage_repo = passage_repo
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True if ``transformers`` and ``torch`` are importable."""
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazily load the tokenizer and causal LM."""
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        _logger.info("loading KV-cache model %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def precompute_text(self, text: str) -> Any:
        """Run one forward pass over *text* and return ``past_key_values``."""
        self._load()
        import torch  # type: ignore

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        with torch.no_grad():
            out = self._model(**inputs, use_cache=True)
        return out.past_key_values

    def precompute(
        self,
        passages: Optional[Iterable[Any]] = None,
        skip_cached: bool = True,
        limit: Optional[int] = None,
    ) -> int:
        """Precompute caches for *passages* (defaults to all repo passages).

        Returns the number of newly cached passages.
        """
        if not self.is_available():
            _logger.warning("transformers/torch unavailable; cannot precompute KV caches")
            return 0
        if passages is None:
            if self.passage_repo is None:
                raise ValueError("no passages provided and no passage_repo wired")
            passages = (
                self.passage_repo.get(pid) for pid in self.passage_repo.all_ids()
            )
        count = 0
        for passage in passages:
            if limit is not None and count >= limit:
                break
            if skip_cached and self.kv_repo.has(passage.id):
                continue
            try:
                cache = self.precompute_text(passage.text)
                n_layers = self._num_layers(cache)
                self.kv_repo.store(
                    passage.id,
                    cache,
                    metadata={
                        "model": self.model_name,
                        "dim": getattr(self._model.config, "hidden_size", None),
                        "layer_count": n_layers,
                    },
                )
                count += 1
            except Exception as exc:  # pragma: no cover
                _logger.warning("failed to cache passage %s: %s", passage.id, exc)
        _logger.info("precomputed KV caches for %d passages", count)
        return count

    @staticmethod
    def _num_layers(cache: Any) -> int:
        """Best-effort count of layers in a KV cache object."""
        try:
            if hasattr(cache, "key_cache"):
                return len(cache.key_cache)
            return len(cache)
        except Exception:  # pragma: no cover
            return 0
