"""Unit tests for entity resolver."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

import pytest

from groundgraph.application.extraction.entity_resolver import EntityResolutionService
from groundgraph.domain.knowledge import CanonicalEntity, EntityMention
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository


class _SpyResolver(EntityResolutionService):
    """Resolver that always returns a fixed similarity score."""

    def __init__(
        self,
        score: float,
        graph_repository: Neo4jGraphRepository | None = None,
        fuzzy_threshold: float = 0.85,
        review_threshold: float = 0.7,
    ) -> None:
        super().__init__(
            graph_repository=graph_repository,
            fuzzy_threshold=fuzzy_threshold,
            review_threshold=review_threshold,
        )
        self._score = score

    def _string_similarity(self, a: str, b: str) -> float:
        return self._score


@pytest.mark.asyncio
async def test_resolve_exact_match() -> None:
    """Exact string match resolves to existing entity."""
    entity = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Database",
        canonical_name="PostgreSQL",
        aliases=["PostgreSQL"],
    )

    class _FakeRepo:
        async def find_entities(self, entity_type: str, limit: int = 20):
            return [entity]

    resolver = EntityResolutionService(
        graph_repository=cast(Neo4jGraphRepository | None, _FakeRepo()), fuzzy_threshold=0.85
    )
    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="PostgreSQL",
        candidate_type="Database",
        extraction_confidence=0.9,
    )

    result = await resolver.resolve(mention)
    assert result is not None
    assert result.canonical_name == "PostgreSQL"


@pytest.mark.asyncio
async def test_resolve_no_match_creates_new() -> None:
    """No matching entity creates a new canonical entity."""

    class _FakeRepo:
        async def find_entities(self, entity_type: str, limit: int = 20):
            return []

        async def create_entity(self, entity: CanonicalEntity) -> CanonicalEntity:
            return entity

    resolver = EntityResolutionService(
        graph_repository=cast(Neo4jGraphRepository | None, _FakeRepo()), fuzzy_threshold=0.85
    )
    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="NewDatabase",
        candidate_type="Database",
        extraction_confidence=0.9,
    )

    result = await resolver.resolve(mention)
    assert result is not None
    assert result.canonical_name == "NewDatabase"
    assert result.entity_type == "Database"


@pytest.mark.asyncio
async def test_resolve_ambiguous_returns_none() -> None:
    """Fuzzy match in review band returns None (needs human review)."""
    entity = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Database",
        canonical_name="PostgreSQL",
        aliases=["PostgreSQL"],
    )

    class _FakeRepo:
        async def find_entities(self, entity_type: str, limit: int = 20):
            return [entity]

        async def create_entity(self, entity: CanonicalEntity) -> CanonicalEntity:
            return entity

    spy = _SpyResolver(
        score=0.8,
        graph_repository=cast(Neo4jGraphRepository | None, _FakeRepo()),
        fuzzy_threshold=0.95,
        review_threshold=0.75,
    )
    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="PostgreSQL_Fork",
        candidate_type="Database",
        extraction_confidence=0.9,
    )

    result = await spy.resolve(mention)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_empty_surface_returns_none() -> None:
    """Empty surface form returns None."""
    resolver = EntityResolutionService()
    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="   ",
        candidate_type="Database",
        extraction_confidence=0.9,
    )

    result = await resolver.resolve(mention)
    assert result is None
