"""Unit tests for LLM-based entity extractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from groundgraph.application.extraction.llm_extractor import (
    LLMEntityExtractor,
    _EntityOutput,
    _ExtractionOutput,
)


class TestEntityOutput:
    """Tests for _EntityOutput validation."""

    def test_entity_output_accepts_valid_data(self) -> None:
        """_EntityOutput validates correct data."""
        entity = _EntityOutput(
            surface_form="AuthService",
            entity_type="Service",
            confidence=0.9,
        )
        assert entity.surface_form == "AuthService"
        assert entity.entity_type == "Service"
        assert entity.confidence == 0.9

    def test_entity_output_allows_confidence_out_of_range(self) -> None:
        """_EntityOutput does not strictly validate confidence range."""
        entity = _EntityOutput(
            surface_form="AuthService",
            entity_type="Service",
            confidence=1.5,
        )
        assert entity.confidence == 1.5


class TestExtractionOutput:
    """Tests for _ExtractionOutput validation."""

    def test_extraction_output_accepts_valid_data(self) -> None:
        """_ExtractionOutput validates entities list."""
        output = _ExtractionOutput(
            entities=[
                _EntityOutput(
                    surface_form="Redis",
                    entity_type="Database",
                    confidence=0.8,
                ),
            ],
        )
        assert len(output.entities) == 1
        assert output.entities[0].surface_form == "Redis"


@pytest.mark.asyncio
async def test_extract_empty_text_returns_empty() -> None:
    """extract() returns [] for empty/whitespace text."""
    extractor = LLMEntityExtractor(client=MagicMock())
    result = await extractor.extract("", uuid4())
    assert result == []


@pytest.mark.asyncio
async def test_extract_whitespace_text_returns_empty() -> None:
    """extract() returns [] for whitespace-only text."""
    extractor = LLMEntityExtractor(client=MagicMock())
    result = await extractor.extract("   \n\t  ", uuid4())
    assert result == []


@pytest.mark.asyncio
async def test_extract_returns_entities_for_valid_response() -> None:
    """extract() returns EntityMentions when LLM returns valid JSON."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"entities":[{"surface_form":"AuthService","entity_type":"Service","confidence":0.9}]}'
            )
        )
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = LLMEntityExtractor(client=mock_client)
    chunk_id = uuid4()
    result = await extractor.extract("AuthService is a service", chunk_id)

    assert len(result) == 1
    assert result[0].surface_form == "AuthService"
    assert result[0].candidate_type == "Service"
    assert result[0].extraction_confidence == 0.9
    assert result[0].chunk_id == chunk_id


@pytest.mark.asyncio
async def test_extract_filters_entities_below_confidence_threshold() -> None:
    """extract() filters out entities with confidence < 0.5."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"entities":[{"surface_form":"UnknownThing","entity_type":"Concept","confidence":0.3}]}'
            )
        )
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = LLMEntityExtractor(client=mock_client)
    result = await extractor.extract("Some text with unknown thing", uuid4())

    assert len(result) == 0


@pytest.mark.asyncio
async def test_extract_returns_empty_on_invalid_json() -> None:
    """extract() returns [] when LLM returns invalid JSON."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = LLMEntityExtractor(client=mock_client)
    result = await extractor.extract("Some text", uuid4())

    assert result == []


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_exception() -> None:
    """extract() returns [] when LLM API raises an exception."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

    extractor = LLMEntityExtractor(client=mock_client)
    result = await extractor.extract("Some text", uuid4())

    assert result == []


@pytest.mark.asyncio
async def test_extract_truncates_long_text_to_4000_chars() -> None:
    """extract() truncates input text to 4000 characters."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"entities":[]}'))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = LLMEntityExtractor(client=mock_client)
    long_text = "x" * 5000
    await extractor.extract(long_text, uuid4())

    call_args = mock_client.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]["content"]
    assert len(user_message) == 4000


@pytest.mark.asyncio
async def test_extract_returns_empty_when_no_entities_in_response() -> None:
    """extract() returns [] when LLM returns empty entities list."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"entities":[]}'))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    extractor = LLMEntityExtractor(client=mock_client)
    result = await extractor.extract("Some text", uuid4())

    assert result == []
