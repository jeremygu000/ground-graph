"""OpenAI embedding provider adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from groundgraph.application.ports import EmbeddingProvider
from groundgraph.application.settings import Settings, get_settings


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    timeout: float = 30.0
    max_retries: int = 3


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI ``text-embedding-3`` family adapter with batching and retry.

    Rate-limit handling follows OpenAI best practices: exponential back-off
    with jitter, capped at ``max_retries`` per batch.  Tokens-per-minute
    limits are respected via the OpenAI SDK's built-in retry mechanism.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = config or EmbeddingConfig()
        settings = settings or get_settings()
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key_value,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        self._model = cfg.model
        self._dimensions = cfg.dimensions
        self._batch_size = cfg.batch_size
        self._max_retries = cfg.max_retries

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch in self._batches(texts):
            batch_embs = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embs)
        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        attempt = 0
        last_exc: Exception | None = None
        delay = 1.0

        while attempt <= self._max_retries:
            try:
                return await self._call_api(texts)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

            attempt += 1

        msg = f"Embedding failed after {self._max_retries + 1} attempts"
        raise RuntimeError(msg) from last_exc

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._supports_dimensions:
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    @property
    def _supports_dimensions(self) -> bool:
        return self._model.startswith("text-embedding-3")

    def _batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]

    async def close(self) -> None:
        await self._client.close()
