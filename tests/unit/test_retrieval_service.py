"""Unit tests for retrieval service."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from groundgraph.application.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)


def _make_fake_embedding_provider(
    model: str = "test-model",
    dimensions: int = 10,
    num_embeddings: int = 2,
) -> MagicMock:
    mock = MagicMock()
    mock.model = model
    mock.dimensions = dimensions
    mock.embed_one = AsyncMock(return_value=[0.1] * dimensions)
    mock.embed = AsyncMock(return_value=[[0.1] * dimensions] * num_embeddings)
    return mock


def _make_config(
    embedding_provider: MagicMock | None = None,
) -> RetrievalServiceConfig:
    return RetrievalServiceConfig(
        session_factory=object(),
        embedding_provider=embedding_provider or _make_fake_embedding_provider(),
        vector_retriever=cast(Any, MagicMock()),
        keyword_retriever=cast(Any, MagicMock()),
        reranker=cast(Any, MagicMock()),
        answer_generator=cast(Any, MagicMock()),
        index_version_resolver=cast(Any, MagicMock()),
    )


@pytest.mark.asyncio
async def test_embed_chunks_returns_embeddings() -> None:
    """embed_chunks returns chunk_id and embedding pairs."""
    chunk1_id = uuid4()
    chunk2_id = uuid4()
    embed_provider = _make_fake_embedding_provider(num_embeddings=2)
    config = _make_config(embedding_provider=embed_provider)
    svc = RetrievalService(config)

    chunks = [
        {"chunk_id": str(chunk1_id), "content": "test content 1"},
        {"chunk_id": str(chunk2_id), "content": "test content 2"},
    ]
    index_version_id = uuid4()

    result = await svc.embed_chunks(chunks, index_version_id)

    assert len(result) == 2
    assert result[0][0] == chunk1_id
    assert result[1][0] == chunk2_id
    embed_provider.embed.assert_called_once()


@pytest.mark.asyncio
async def test_embed_chunks_empty_list_returns_empty() -> None:
    """embed_chunks with empty list returns empty list."""
    embed_provider = _make_fake_embedding_provider(num_embeddings=0)
    config = _make_config(embedding_provider=embed_provider)
    svc = RetrievalService(config)

    result = await svc.embed_chunks([], uuid4())

    assert result == []
    embed_provider.embed.assert_called_once_with([])


@pytest.mark.asyncio
async def test_retrieval_service_init_with_config() -> None:
    """RetrievalService initializes correctly with provided config."""
    embed_provider = _make_fake_embedding_provider()
    config = _make_config(embedding_provider=embed_provider)

    svc = RetrievalService(config)

    assert svc._embed is embed_provider


@pytest.mark.asyncio
async def test_retrieval_service_query_requires_tenant_id() -> None:
    """query() raises ValueError when tenant_id is empty string."""
    config = _make_config()
    svc = RetrievalService(config)

    with pytest.raises(ValueError, match="tenant_id"):
        await svc.query(question="test", principal="user1", tenant_id="")
