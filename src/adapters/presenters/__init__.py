"""Presenter adapters: format use-case outputs for display/transport.

Presenters sit at the adapter boundary and turn application DTOs into shapes
suitable for the Gradio UI, CLI, or API without leaking domain internals.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.domain.value_objects import RetrievalResult


class QueryPresenter:
    """Formats a query result into a display-friendly dictionary."""

    @staticmethod
    def present_result(
        answer: str,
        results: Sequence[RetrievalResult],
        confidence: float,
        compression_ratio: float,
        full_tokens: int,
        compressed_tokens: int,
        query_id: str,
    ) -> Dict[str, Any]:
        """Return a structured view of a query answer for the UI/CLI."""
        return {
            "query_id": query_id,
            "answer": answer,
            "confidence": round(float(confidence), 4),
            "compression_ratio": round(float(compression_ratio), 4),
            "token_reduction": round(max(0.0, 1.0 - compressed_tokens / max(1, full_tokens)), 4),
            "full_tokens": int(full_tokens),
            "compressed_tokens": int(compressed_tokens),
            "sources": [QueryPresenter.present_source(r) for r in results],
        }

    @staticmethod
    def present_source(result: RetrievalResult) -> Dict[str, Any]:
        """Return a compact view of a single retrieval source."""
        return {
            "passage_id": result.passage_id,
            "score": round(float(result.final_score), 4),
            "preview": (result.text[:200] + "…") if len(result.text) > 200 else result.text,
            "source": result.source,
        }


__all__ = ["QueryPresenter"]
