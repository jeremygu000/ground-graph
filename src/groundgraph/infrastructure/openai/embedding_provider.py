"""OpenAI embedding provider adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer

from groundgraph.application.ports import EmbeddingProvider
from groundgraph.application.settings import Settings, get_settings

_TRACER = get_tracer(__name__)
_METER = get_meter(__name__)

_EMBED_DURATION = _METER.create_histogram(
    "groundgraph.retrieval.embedding.duration",
    description="Embedding call duration in milliseconds.",
    unit="ms",
)
_EMBED_TOKENS = _METER.create_counter(
    "groundgraph.retrieval.embedding.tokens",
    description="Total embedding tokens consumed.",
)


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
        settings = settings or get_settings()
        cfg = config or EmbeddingConfig(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
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

        with _TRACER.start_as_current_span("embedding.call_api") as span:
            span.set_attribute("embedding.batch_size", len(texts))
            span.set_attribute("embedding.model", self._model)
            start = time.perf_counter()
            response = await self._client.embeddings.create(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            _EMBED_DURATION.record(duration_ms)
            if response.usage:
                _EMBED_TOKENS.add(response.usage.total_tokens)
                span.set_attribute("embedding.tokens", response.usage.total_tokens)
            span.set_attribute("embedding.duration_ms", duration_ms)
            return [item.embedding for item in response.data]

    @property
    def _supports_dimensions(self) -> bool:
        return self._model.startswith("text-embedding-3")

    def _batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]

    async def close(self) -> None:
        await self._client.close()
