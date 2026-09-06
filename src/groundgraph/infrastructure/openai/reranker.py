"""LLM-based evidence reranker using cross-encoder scoring."""

from __future__ import annotations

import time

from openai import AsyncOpenAI
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer

from groundgraph.application.ports import EvidenceReranker
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import Evidence

_TRACER = get_tracer(__name__)
_METER = get_meter(__name__)

_RERANK_DURATION = _METER.create_histogram(
    "groundgraph.retrieval.rerank.duration",
    description="Reranking duration in milliseconds.",
    unit="ms",
)
_RERANK_TOKENS = _METER.create_counter(
    "groundgraph.retrieval.rerank.tokens",
    description="Total reranking tokens consumed.",
)
_RERANK_EVIDENCE = _METER.create_histogram(
    "groundgraph.retrieval.rerank.evidence_count",
    description="Number of evidence chunks sent to reranker.",
)


class CrossEncoderReranker(EvidenceReranker):
    """Rerank evidence using an LLM with relevance scoring.

    This uses a simple prompt-based relevance scorer rather than a dedicated
    cross-encoder model, since the primary use case is a domain-specific
    knowledge assistant where LLM judgment is preferred.

    Reranking is done by scoring each evidence chunk against the query
    using a 1-5 relevance scale, then sorting descending.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._client = client or AsyncOpenAI(api_key=cfg.openai_api_key_value)
        self._model = model or cfg.generation_model

    async def rerank(self, evidence: list[Evidence], query: str) -> list[Evidence]:
        if len(evidence) <= 1:
            return list(evidence)

        with _TRACER.start_as_current_span("rerank.score_chunks") as span:
            span.set_attribute("rerank.evidence_count", len(evidence))
            span.set_attribute("rerank.query_length", len(query))
            start = time.perf_counter()
            scored: list[tuple[Evidence, float]] = []
            total_tokens = 0
            for ev in evidence:
                score, tokens = await self._score_relevance(query, ev.content)
                scored.append((ev, score))
                total_tokens += tokens

            duration_ms = (time.perf_counter() - start) * 1000
            _RERANK_DURATION.record(duration_ms)
            _RERANK_TOKENS.add(total_tokens)
            _RERANK_EVIDENCE.record(len(evidence))
            span.set_attribute("rerank.duration_ms", duration_ms)
            span.set_attribute("rerank.tokens", total_tokens)

            scored.sort(key=lambda x: x[1], reverse=True)
            span.set_attribute("rerank.scored_count", len(scored))
            reranked = []
            for ev, _ in scored:
                reranked.append(
                    ev.model_copy(
                        update={
                            "rerank_score": _,
                        }
                    )
                )
            return reranked

    async def _score_relevance(self, query: str, content: str) -> tuple[float, int]:
        prompt = (
            "You are a relevance assessor. Rate how well the following evidence chunk "
            "answers the user's question on a scale of 1 to 5.\n\n"
            f"Question: {query}\n\n"
            f"Evidence: {content[:2000]}\n\n"
            "Respond with a single number between 1 and 5 (1=irrelevant, 5=perfectly relevant)."
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )

        text = response.choices[0].message.content or "1"
        tokens = response.usage.total_tokens if response.usage else 0
        try:
            score = float(text.strip().split()[0])
            return max(1.0, min(5.0, score)), tokens
        except (ValueError, IndexError):
            return 1.0, tokens

    async def close(self) -> None:
        await self._client.close()
