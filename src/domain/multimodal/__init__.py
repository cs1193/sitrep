"""Multimodal domain entities (Phase G1/G2): images, audio, video, cross-modal links."""
from src.domain.multimodal.image_entities import CrossModalLink, ImageEntity
from src.domain.multimodal.av_entities import AudioEntity, TemporalSegment, VideoEntity

__all__ = [
    "ImageEntity",
    "CrossModalLink",
    "AudioEntity",
    "VideoEntity",
    "TemporalSegment",
]
