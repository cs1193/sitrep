"""Audio/video embedder (Phase G2).

When ``transformers``/``torch`` are available, transcribes + embeds via a real
model. Otherwise falls back to **transcript-based hashing** so audio/video assets
embed in the same text space as queries — functional with zero model downloads.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.utils.constants import EMBEDDING_DIM

_logger = logging.getLogger("sitrep.embedding.av")


class AVEmbedder:
    """Embeds audio/video assets via their transcript (lazy model, hash fallback)."""

    name = "av"

    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        fallback_dim: int = EMBEDDING_DIM,
        device: str = "cpu",
    ) -> None:
        """Configure the ASR/embedding model name, fallback dimensionality, device."""
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.device = device
        self._model: Any = None

    def is_available(self) -> bool:
        """Return True if ``transformers`` and ``torch`` are importable."""
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def embed_transcript(self, transcript: str) -> List[float]:
        """Embed a *transcript* (the textual proxy for an audio/video asset)."""
        if self.is_available():
            try:
                from src.infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedder

                emb = SentenceTransformerEmbedder(dim=self.fallback_dim)
                if emb.is_available():
                    emb._load()  # type: ignore[operator]
                    return emb.embed(transcript)
            except Exception as exc:  # pragma: no cover
                _logger.warning("AV real embed failed, using hash: %s", exc)
        from src.utils.embedding import hash_embedding

        return hash_embedding(transcript or "", self.fallback_dim)

    def embed_audio(self, transcript: str, audio_path: Optional[str] = None) -> List[float]:
        """Embed an audio asset: transcribe *audio_path* if given, else use *transcript*."""
        if audio_path and self.is_available():
            try:
                import torch  # type: ignore
                from transformers import WhisperProcessor, WhisperForConditionalGeneration  # type: ignore

                import librosa  # type: ignore

                _logger.info("transcribing %s with %s", audio_path, self.model_name)
                processor = WhisperProcessor.from_pretrained(self.model_name)
                model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                speech, _sr = librosa.load(audio_path, sr=16000)
                inputs = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
                with torch.no_grad():
                    ids = model.generate(inputs, forced_decoder_ids=forced)
                transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            except Exception as exc:  # pragma: no cover
                _logger.warning("AV transcription failed, using provided transcript: %s", exc)
        return self.embed_transcript(transcript)
