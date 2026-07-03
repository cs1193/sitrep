"""Controller adapters: thin orchestration hooks for entry points (CLI/web/API).

Controllers translate external requests into use-case calls and route the
result to a presenter. Kept deliberately thin; the bulk of logic lives in the
application use cases.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class QueryController:
    """Bridges an inbound query request to :class:`QueryKnowledgeUseCase`."""

    def __init__(self, use_case: Any) -> None:
        """Wire the query use case."""
        self.use_case = use_case

    def handle(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute the query and return a presenter-formatted result."""
        result = self.use_case.execute(query=query, top_k=top_k)
        if hasattr(self.use_case, "presenter") and self.use_case.presenter is not None:
            return self.use_case.presenter(result)
        return result


class IngestController:
    """Bridges an inbound ingest request to :class:`IngestDocumentUseCase`."""

    def __init__(self, use_case: Any) -> None:
        """Wire the ingest use case."""
        self.use_case = use_case

    def handle(self, text: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute ingestion and return a summary dict."""
        return self.use_case.execute(text=text, document_id=document_id)


__all__ = ["QueryController", "IngestController"]
