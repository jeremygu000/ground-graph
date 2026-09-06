"""Unit tests for deterministic entity extractor."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.application.extraction.deterministic_extractor import DeterministicEntityExtractor


@pytest.mark.asyncio
async def test_extracts_database_engines() -> None:
    """PostgreSQL and Redis are extracted as Database entities."""
    extractor = DeterministicEntityExtractor()
    text = "We use PostgreSQL for the primary database and Redis for caching."
    chunk_id = uuid4()

    mentions = await extractor.extract(text, chunk_id)

    types = {m.candidate_type for m in mentions}
    assert "Database" in types
    surfaces = {m.surface_form for m in mentions}
    assert "PostgreSQL" in surfaces
    assert "Redis" in surfaces


@pytest.mark.asyncio
async def test_extracts_env_variables() -> None:
    """ENV variables are extracted as Configuration entities."""
    extractor = DeterministicEntityExtractor()
    text = "Set DATABASE_URL and API_KEY in the environment."
    chunk_id = uuid4()

    mentions = await extractor.extract(text, chunk_id)

    types = {m.candidate_type for m in mentions}
    assert "Configuration" in types


@pytest.mark.asyncio
async def test_extracts_github_urls() -> None:
    """GitHub URLs are extracted as Repository entities."""
    extractor = DeterministicEntityExtractor()
    text = "The code lives at https://github.com/jeremygu000/ground-graph"
    chunk_id = uuid4()

    mentions = await extractor.extract(text, chunk_id)

    types = {m.candidate_type for m in mentions}
    assert "Repository" in types


@pytest.mark.asyncio
async def test_empty_text_returns_empty() -> None:
    """Empty text yields no mentions."""
    extractor = DeterministicEntityExtractor()
    mentions = await extractor.extract("", uuid4())
    assert mentions == []


@pytest.mark.asyncio
async def test_no_conflicts_returns_empty() -> None:
    """Text with no known patterns returns empty list."""
    extractor = DeterministicEntityExtractor()
    mentions = await extractor.extract("This is just prose with no entities.", uuid4())
    assert mentions == []
