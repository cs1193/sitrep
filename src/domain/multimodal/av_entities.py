"""Audio / video entities with temporal segmentation (Phase G2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.common import generate_id, utc_now_iso


@dataclass
class TemporalSegment:
    """A time-bounded segment of an audio/video asset (with optional text)."""

    start: float
    end: float
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("TemporalSegment.end must be >= start")

    @property
    def duration(self) -> float:
        """Return the segment duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text, "metadata": dict(self.metadata)}


@dataclass
class AudioEntity:
    """An audio asset with a transcript and temporal segments."""

    source: str = ""
    transcript: str = ""
    segments: List[TemporalSegment] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    linked_passage_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("audio"))
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class VideoEntity:
    """A video asset combining visual + audio tracks with temporal segments."""

    source: str = ""
    transcript: str = ""
    segments: List[TemporalSegment] = field(default_factory=list)
    visual_embedding: Optional[List[float]] = None
    audio_embedding: Optional[List[float]] = None
    linked_passage_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: generate_id("video"))
    created_at: str = field(default_factory=utc_now_iso)
