"""Extraction service: pulls schemas and facts from a passage.

Provides a deterministic regex-based extractor (works with no LLM) and an
LLM-assisted path used when a non-demo gateway is available. The regex patterns
are intentionally simple for demo purposes; production would rely on the LLM
prompts.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.domain.schemas import Fact, Schema
from src.domain.interfaces import LLMGateway
from src.utils.common import generate_id, truncate

_logger = logging.getLogger("sitrep.services.extraction")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# (compiled_pattern, predicate) — first match wins per sentence.
_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+(.+)", re.I), "is_a"),
    (re.compile(r"(.+?)\s+(?:is|are|was|were)\s+(.+)", re.I), "is"),
    (re.compile(r"(.+?)\s+(?:has|have|contains|includes)\s+(.+)", re.I), "has"),
    (re.compile(r"(.+?)\s+(?:located in|based in|headquartered in)\s+(.+)", re.I), "located_in"),
    (re.compile(r"(.+?)\s+(?:born in|founded in|established in|created in)\s+(.+)", re.I), "founded_in"),
    (re.compile(r"(.+?)\s+(?:equals|means|refers to)\s+(.+)", re.I), "means"),
]

_KV_LINE = re.compile(r"^([A-Za-z][\w \-]{1,40}):\s*(.+)$")


@dataclass
class ExtractionResult:
    """Outcome of extraction: an inferred schema and zero or more facts."""

    schema: Optional[Schema] = None
    facts: List[Fact] = field(default_factory=list)
    method: str = "regex"


class ExtractionService:
    """Extracts structured facts and an inferred schema from raw text."""

    def __init__(self, llm_gateway: Optional[LLMGateway] = None, domain: str = "general") -> None:
        """Store the optional LLM gateway and default domain label."""
        self.llm = llm_gateway
        self.domain = domain

    # ----------------------------------------------------------------- public API
    def extract(self, text: str, source_passage_id: Optional[str] = None) -> ExtractionResult:
        """Extract facts from *text* (LLM-assisted when possible, else regex)."""
        if not text or not text.strip():
            return ExtractionResult()
        if self._can_use_llm():
            try:
                result = self._extract_with_llm(text, source_passage_id)
                if result.facts:
                    return result
            except Exception as exc:  # pragma: no cover
                _logger.warning("LLM extraction failed, falling back to regex: %s", exc)
        return self._extract_regex(text, source_passage_id)

    # ----------------------------------------------------------------- regex path
    def _extract_regex(self, text: str, source_passage_id: Optional[str]) -> ExtractionResult:
        """Deterministic regex extraction."""
        facts: List[Fact] = []
        seen = set()
        # Key-value lines first (high precision).
        for line in (text or "").splitlines():
            m = _KV_LINE.match(line.strip())
            if m:
                self._add_fact(facts, seen, m.group(1).strip(), "has", m.group(2).strip(),
                               source_passage_id)
        # Sentence patterns.
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 6:
                continue
            for pattern, predicate in _PATTERNS:
                m = pattern.match(sentence)
                if m:
                    subject = self._clean_phrase(m.group(1))
                    obj = self._clean_phrase(m.group(2))
                    if subject and obj:
                        self._add_fact(facts, seen, subject, predicate, obj, source_passage_id)
                    break
        schema = self._infer_schema(facts)
        return ExtractionResult(schema=schema, facts=facts, method="regex")

    @staticmethod
    def _clean_phrase(phrase: str) -> Optional[str]:
        """Trim and validate a subject/object phrase."""
        p = re.sub(r"\s+", " ", (phrase or "").strip().rstrip(".;,"))

        def _strip_leading(p: str) -> str:
            return re.sub(r"^(?:and|but|the|a|an|so|then|which|that)\s+", "", p, flags=re.I).strip()

        p = _strip_leading(p)
        if not (2 <= len(p) <= 160):
            return None
        # Numeric-leading phrases are valid facts (prices, quantities, years, measures).
        return p

    @staticmethod
    def _add_fact(facts, seen, subject, predicate, obj, passage_id) -> None:
        """Append a fact if its (subject, predicate, object) is novel."""
        key = (subject.lower(), predicate, obj.lower())
        if key in seen:
            return
        seen.add(key)
        passages = [passage_id] if passage_id else []
        facts.append(
            Fact(
                subject=subject,
                predicate=predicate,
                object_value=obj,
                source_passage_ids=passages,
                confidence=0.6,
            )
        )

    def _infer_schema(self, facts: List[Fact]) -> Optional[Schema]:
        """Derive a :class:`Schema` from the extracted predicates."""
        if not facts:
            return None
        predicates = []
        for f in facts:
            if f.predicate not in predicates:
                predicates.append(f.predicate)
        fields = [{"name": "subject", "type": "entity"}] + [
            {"name": p, "type": "text"} for p in predicates
        ]
        schema = Schema(
            name=f"{self.domain}_facts",
            description=f"Inferred schema for {self.domain} facts",
            fields=fields,
            domain=self.domain,
        )
        schema.usage_count = len(facts)
        return schema

    # ----------------------------------------------------------------- LLM path
    def _can_use_llm(self) -> bool:
        """Return True if a non-demo LLM gateway is configured."""
        return self.llm is not None and getattr(self.llm, "name", "") != "demo"

    def _extract_with_llm(self, text: str, source_passage_id: Optional[str]) -> ExtractionResult:
        """Ask the LLM for JSON facts; fall back to regex on parse failure."""
        prompt = (
            "Extract factual triples from the text. Respond with ONLY JSON:\n"
            '{"schema":"<name>","facts":[{"subject":"...","predicate":"...","object":"..."}]}\n'
            f"Text:\n{truncate(text, 3000)}"
        )
        raw = self.llm.generate(prompt)
        data = self._parse_json(raw)
        facts: List[Fact] = []
        seen = set()
        for item in data.get("facts", []):
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "is")).strip() or "is"
            obj = str(item.get("object", "")).strip()
            if subject and obj:
                self._add_fact(facts, seen, subject, predicate, obj, source_passage_id)
        schema = self._infer_schema(facts) or Schema(
            name=str(data.get("schema") or f"{self.domain}_facts"), domain=self.domain
        )
        return ExtractionResult(schema=schema, facts=facts, method="llm")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Best-effort extraction of a JSON object from an LLM response."""
        if not raw:
            return {}
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
