"""Unit tests for the OpenAI embedding provider."""

from __future__ import annotations

from typing import Any

import pytest

from groundgraph.infrastructure.openai.embedding_provider import (
    EmbeddingConfig,
    OpenAIEmbeddingProvider,
)


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeCreateResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(e) for e in embeddings]
        self.usage = _FakeUsage()


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.total_tokens = 10


class _FakeOpenAIClient:
    def __init__(self, embeddings: list[list[float]] | Exception) -> None:
        self._embeddings = embeddings
        self.embeddings = _FakeEmbeddingsNamespace(embeddings)

    async def close(self) -> None:
        pass


class _FakeEmbeddingsNamespace:
    def __init__(self, embeddings: list[list[float]] | Exception) -> None:
        self._embeddings = embeddings

    async def create(self, **kwargs: Any) -> _FakeCreateResponse:
        if isinstance(self._embeddings, Exception):
            raise self._embeddings
        return _FakeCreateResponse(self._embeddings)


class TestEmbeddingConfig:
    def test_default_values(self) -> None:
        config = EmbeddingConfig()
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.timeout == 30.0
        assert config.max_retries == 3


class TestOpenAIEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_embed_one_returns_embedding(self) -> None:
        provider = OpenAIEmbeddingProvider(
            client=_FakeOpenAIClient([[0.1, 0.2, 0.3]]),  # type: ignore[arg-type]
        )
        result = await provider.embed_one("hello world")
        assert result == [0.1, 0.2, 0.3]
        assert provider.model == "text-embedding-3-small"
        assert provider.dimensions == 1536

    @pytest.mark.asyncio
    async def test_embed_batches_large_input(self) -> None:
        call_count = [0]

        class _CountingClient:
            def __init__(self) -> None:
                pass

            async def close(self) -> None:
                pass

        class _CountingEmbeddingsNamespace:
            def __init__(self) -> None:
                self._count = call_count

            async def create(self, **kwargs: Any) -> _FakeCreateResponse:
                count = len(kwargs.get("input", []))
                self._count[0] = count
                emb = [[0.1, 0.2, 0.3] for _ in range(count)]
                return _FakeCreateResponse(emb)

        class _CountingOpenAIClient:
            def __init__(self) -> None:
                self.embeddings = _CountingEmbeddingsNamespace()

            async def close(self) -> None:
                pass

        provider = OpenAIEmbeddingProvider(client=_CountingOpenAIClient())  # type: ignore[arg-type]
        texts = ["text1", "text2", "text3"]
        result = await provider.embed(texts)
        assert len(result) == 3
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_embed_empty_list_returns_empty(self) -> None:
        provider = OpenAIEmbeddingProvider(
            client=_FakeOpenAIClient([]),  # type: ignore[arg-type]
        )
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_retry_on_exception(self) -> None:
        class _ErrorClient:
            def __init__(self) -> None:
                pass

            async def close(self) -> None:
                pass

        class _ErrorEmbeddingsNamespace:
            async def create(self, **kwargs: Any) -> _FakeCreateResponse:
                raise ValueError("test error")

        class _ErrorOpenAIClient:
            embeddings = _ErrorEmbeddingsNamespace()

            async def close(self) -> None:
                pass

        provider = OpenAIEmbeddingProvider(
            client=_ErrorOpenAIClient(),  # type: ignore[arg-type]
            config=EmbeddingConfig(max_retries=2),
        )

        with pytest.raises(RuntimeError, match="Embedding failed after 3 attempts"):
            await provider.embed(["test"])
