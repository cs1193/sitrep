"""CLIP embedder for cross-modal (image ↔ text) retrieval (Phase G1).

When ``transformers`` + ``torch`` + an image are available, uses the real CLIP
model. Otherwise falls back to **caption-based hashing** so an image is embedded
in the same text space as queries — cross-modal retrieval stays functional with
zero model downloads.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.utils.constants import EMBEDDING_DIM

_logger = logging.getLogger("sitrep.embedding.clip")


class CLIPEmbedder:
    """Image + text embedder (lazy CLIP, caption-hash fallback)."""

    name = "clip"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        fallback_dim: int = EMBEDDING_DIM,
        device: str = "cpu",
    ) -> None:
        """Configure the CLIP model name, fallback dimensionality, and device."""
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.device = device
        self._model: Any = None
        self._processor: Any = None
        self._projection_dim: Optional[int] = None

    def is_available(self) -> bool:
        """Return True if ``transformers`` and ``torch`` are importable."""
        try:
            import torch  # type: ignore  # noqa: F401
            import transformers  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        """Lazily load the CLIP model + processor."""
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        _logger.info("loading CLIP %s on %s", self.model_name, self.device)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        try:
            self._projection_dim = int(self._model.config.projection_dim)
        except Exception:  # pragma: no cover
            self._projection_dim = self.fallback_dim

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality (CLIP projection dim or fallback)."""
        return self._projection_dim or self.fallback_dim

    def embed_text(self, text: str) -> List[float]:
        """Embed *text* into the shared cross-modal space."""
        if self.is_available():
            try:
                self._load()
                import torch  # type: ignore

                inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    feats = self._model.get_text_features(**inputs)
                import numpy as np  # type: ignore

                import src.utils.common as _c
                return _c.normalize(feats[0].cpu().double().numpy().tolist())
            except Exception as exc:  # pragma: no cover
                _logger.warning("CLIP text encode failed, using fallback: %s", exc)
        from src.utils.embedding import hash_embedding

        return hash_embedding(text, self.fallback_dim)

    def embed_image(self, caption: str, image_path: Optional[str] = None) -> List[float]:
        """Embed an image; fall back to its *caption* when no image/CLIP is present."""
        if image_path and self.is_available():
            try:
                self._load()
                from PIL import Image  # type: ignore
                import torch  # type: ignore

                image = Image.open(image_path).convert("RGB")
                inputs = self._processor(images=[image], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = self._model.get_image_features(**inputs)
                import src.utils.common as _c

                return _c.normalize(feats[0].cpu().double().numpy().tolist())
            except Exception as exc:  # pragma: no cover
                _logger.warning("CLIP image encode failed, using caption fallback: %s", exc)
        return self.embed_text(caption)
