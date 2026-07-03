"""HuggingFace Transformers LLM client (extra ``[llm]``)."""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.domain.interfaces import LLMGateway

_logger = logging.getLogger("sitrep.llm.transformers")


class TransformersLLMClient(LLMGateway):
    """In-process causal-LM client via ``transformers`` (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        device: str = "cpu",
    ) -> None:
        """Store config; the model loads on first ``generate`` call."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.name = "transformers"
        self._tokenizer: Any = None
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True if both ``torch`` and ``transformers`` import."""
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazily load tokenizer + model onto the configured device."""
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        _logger.info("loading LLM %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a completion for *prompt* using chat formatting when available."""
        self._load()
        import torch  # type: ignore

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = ((system + "\n") if system else "") + prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        temperature = float(kwargs.get("temperature", self.temperature))
        max_new = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
