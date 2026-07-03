"""Multimodal use case (Phase G1/G2): ingest images/audio + cross-modal retrieval."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.adapters.repositories.multimodal_repo import SQLiteMultimodalRepo
from src.domain.multimodal.av_entities import AudioEntity, TemporalSegment, VideoEntity
from src.domain.multimodal.image_entities import CrossModalLink, ImageEntity
from src.infrastructure.embedding.av import AVEmbedder
from src.infrastructure.embedding.clip import CLIPEmbedder
from src.utils.common import cosine_similarity

_logger = logging.getLogger("sitrep.usecase.multimodal")


class MultimodalUseCase:
    """Ingest multimodal assets and retrieve them with cross-modal links."""

    def __init__(
        self,
        mm_repo: SQLiteMultimodalRepo,
        clip_embedder: CLIPEmbedder,
        av_embedder: Optional[AVEmbedder] = None,
    ) -> None:
        """Wire the multimodal repo + CLIP/AV embedders."""
        self.mm_repo = mm_repo
        self.clip_embedder = clip_embedder
        self.av_embedder = av_embedder

    def ingest_image(
        self,
        caption: str,
        linked_passage_ids: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        """Embed + store an image, linking it to *linked_passage_ids*."""
        embedding = self.clip_embedder.embed_image(caption, image_path=image_path)
        image = ImageEntity(
            caption=caption,
            source=source,
            visual_embedding=embedding,
            linked_passage_ids=list(linked_passage_ids or []),
        )
        self.mm_repo.save_image(image)
        for passage_id in linked_passage_ids or []:
            self.mm_repo.save_link(CrossModalLink(image_id=image.id, passage_id=passage_id))
        _logger.info("ingested image %s (caption=%r)", image.id, caption[:40])
        return {"image_id": image.id, "caption": caption, "linked_passages": list(linked_passage_ids or [])}

    def ingest_audio(
        self,
        transcript: str,
        segments: Optional[List[TemporalSegment]] = None,
        source: str = "",
        linked_passage_ids: Optional[List[str]] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Embed (via transcript) + store an audio asset."""
        embedding = self.av_embedder.embed_audio(transcript, audio_path=audio_path) if self.av_embedder else []
        audio = AudioEntity(
            source=source,
            transcript=transcript,
            segments=list(segments or []),
            embedding=embedding,
            linked_passage_ids=list(linked_passage_ids or []),
        )
        self.mm_repo.save_media(audio)
        return {"audio_id": audio.id, "transcript": transcript, "segments": len(segments or [])}

    def ingest_video(
        self,
        transcript: str,
        segments: Optional[List[TemporalSegment]] = None,
        source: str = "",
        linked_passage_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Embed (via transcript) + store a video asset."""
        embedding = self.av_embedder.embed_transcript(transcript) if self.av_embedder else []
        video = VideoEntity(
            source=source,
            transcript=transcript,
            segments=list(segments or []),
            visual_embedding=embedding,
            linked_passage_ids=list(linked_passage_ids or []),
        )
        self.mm_repo.save_media(video)
        return {"video_id": video.id, "transcript": transcript, "segments": len(segments or [])}

    def retrieve_cross_modal(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Find images whose caption matches *query* (cross-modal), with linked passages."""
        query_emb = self.clip_embedder.embed_text(query)
        scored = []
        for image in self.mm_repo.iter_images_with_embeddings():
            if not image.visual_embedding:
                continue
            sim = cosine_similarity(query_emb, image.visual_embedding)
            scored.append((image, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [
            {
                "image_id": image.id,
                "caption": image.caption,
                "score": round(sim, 4),
                "linked_passages": list(image.linked_passage_ids),
            }
            for image, sim in scored[:top_k]
        ]
        return {"query": query, "images": results}
