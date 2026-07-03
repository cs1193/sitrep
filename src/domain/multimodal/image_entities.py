"""Image + cross-modal-link entities (Phase G1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.common import generate_id, utc_now_iso


@dataclass
class ImageEntity:
    """An image asset with an optional caption and visual embedding."""

    caption: str = ""
    source: str = ""
    visual_embedding: Optional[List[float]] = None
    linked_passage_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("image"))
    created_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if self.caption is None:
            self.caption = ""

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "caption": self.caption,
            "source": self.source,
            "linked_passage_ids": list(self.linked_passage_ids),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }
        if include_embedding:
            data["visual_embedding"] = list(self.visual_embedding) if self.visual_embedding else None
        return data


@dataclass
class CrossModalLink:
    """A typed link between an image and a text passage (e.g. 'depicts')."""

    image_id: str
    passage_id: str
    kind: str = "depicts"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("xlink"))
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "passage_id": self.passage_id,
            "kind": self.kind,
            "weight": self.weight,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }
