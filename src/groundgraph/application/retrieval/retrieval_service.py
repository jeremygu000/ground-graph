"""Hybrid retrieval service combining vector + keyword search with RRF."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from groundgraph.application.ports import AnswerGenerator, EmbeddingProvider, EvidenceReranker
from groundgraph.application.retrieval.fusion import reciprocal_rank_fusion
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan
from groundgraph.infrastructure.postgres.keyword_retriever import (
    PostgresKeywordRetriever,
)
from groundgraph.infrastructure.postgres.vector_retriever import PostgresVectorRetriever


class RetrievalService:
    """Orchestrate hybrid vector + keyword retrieval, reranking, and answer generation."""

    def __init__(
        self,
        session_factory: Any,
        embedding_provider: EmbeddingProvider,
        reranker: EvidenceReranker,
        answer_generator: AnswerGenerator,
        settings: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embed = embedding_provider
        self._reranker = reranker
        self._answer_generator = answer_generator
        self._settings = settings or get_settings()

    async def query(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        vector_top_k: int | None = None,
        final_limit: int | None = None,
    ) -> QueryResponse:
        vector_top_k = vector_top_k or self._settings.vector_top_k
        final_limit = final_limit or self._settings.final_evidence_limit

        async with self._session_factory() as session:
            vector_retriever = PostgresVectorRetriever(session)
            keyword_retriever = PostgresKeywordRetriever(session)

            query_vector = await self._embed.embed_one(question)

            vector_results = await vector_retriever.search_with_content(
                query_vector,
                top_k=vector_top_k,
                filters={"allowed_principals": [principal]} if principal else None,
            )

            keyword_results = await keyword_retriever.search(
                question,
                top_k=self._settings.keyword_top_k,
                allowed_principals=[principal],
            )

        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        top_results = fused[:final_limit]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=r.source_id,
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                content=r.content,
                retrieval_method="vector" if r.vector_score else "keyword",
                vector_score=r.vector_score,
                rerank_score=None,
                allowed_principals=r.allowed_principals,
            )
            for r in top_results
        ]

        reranked = await self._reranker.rerank(evidence, question)

        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=[question],
            vector_top_k=vector_top_k,
            final_evidence_limit=final_limit,
        )

        return await self._answer_generator.generate(question, reranked, plan)

    async def embed_chunks(
        self,
        chunks: list[dict[str, Any]],
        index_version_id: UUID,
    ) -> list[tuple[UUID, list[float]]]:
        texts = [c["content"] for c in chunks]
        embeddings = await self._embed.embed(texts)
        return [(UUID(c["chunk_id"]), emb) for c, emb in zip(chunks, embeddings, strict=True)]
