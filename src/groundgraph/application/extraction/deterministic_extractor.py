"""Ontology-aware entity extractor using deterministic rules."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID, uuid4

from groundgraph.domain.knowledge import EntityMention
from groundgraph.domain.ontology.loader import get_ontology

if TYPE_CHECKING:
    from groundgraph.domain.ontology.loader import Ontology

_MIN_SURFACE_LEN = 2


class DeterministicEntityExtractor:
    """Rule-based entity extractor for code/config entities.

    Extracts entities by matching regex patterns against text.
    Covers:
      - Software systems (product/service names)
      - Repositories (GitHub URLs, internal paths)
      - Databases (engine names, connection strings)
      - Configuration keys (env vars, config files)
    """

    _PATTERNS: ClassVar[list[tuple[str, str, re.Pattern[str]]]] = [
        (
            "SoftwareSystem",
            "product name (capitalized)",
            re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:DB|API|SDK|CLI|UI|Service)\b"),
        ),
        (
            "Database",
            "PostgreSQL/MySQL/MongoDB/etc",
            re.compile(r"\b(PostgreSQL|MySQL|MongoDB|Redis|Neo4j|Pgvector|Elasticsearch)\b"),
        ),
        (
            "Service",
            "HTTP endpoint/path",
            re.compile(r"\b(?:https?://|/api/)[^\s'\"<>]+"),
        ),
        (
            "Configuration",
            "ENV variable",
            re.compile(r"\b[A-Z][A-Z0-9_]*(?:_URL|_HOST|_PORT|_KEY|_SECRET|_TOKEN)\b"),
        ),
        (
            "Repository",
            "GitHub URL",
            re.compile(r"https://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+"),
        ),
    ]

    def __init__(self, ontology: Ontology | None = None) -> None:
        self._ontology = ontology or get_ontology()

    async def extract(self, text: str, chunk_id: UUID) -> list[EntityMention]:
        """Extract deterministic entities from *text*."""
        results: list[EntityMention] = []
        for entity_type, _, pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                surface = match.group()
                if len(surface) < _MIN_SURFACE_LEN:
                    continue
                if not self._ontology.is_valid_entity_type(entity_type):
                    continue
                mention = EntityMention(
                    mention_id=uuid4(),
                    chunk_id=chunk_id,
                    surface_form=surface,
                    candidate_type=entity_type,
                    locator=match.group(),
                    extraction_confidence=0.9,
                )
                results.append(mention)
        return results
