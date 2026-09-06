"""LLM-based evidence reranker using cross-encoder scoring."""

from __future__ import annotations

from openai import AsyncOpenAI
from opentelemetry.trace import get_tracer

from groundgraph.application.ports import EvidenceReranker
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import Evidence

_TRACER = get_tracer(__name__)


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
            scored: list[tuple[Evidence, float]] = []
            for ev in evidence:
                score = await self._score_relevance(query, ev.content)
                scored.append((ev, score))

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

    async def _score_relevance(self, query: str, content: str) -> float:
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
        try:
            score = float(text.strip().split()[0])
            return max(1.0, min(5.0, score))
        except (ValueError, IndexError):
            return 1.0

    async def close(self) -> None:
        await self._client.close()
