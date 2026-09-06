"""Hybrid retrieval service combining vector + keyword search with RRF.

Application-layer service.  Depends only on application ports; the
infrastructure layer provides concrete ``VectorContentRetriever`` and
``KeywordRetrieverPort`` implementations (see
``groundgraph.infrastructure.composition``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select

from groundgraph.application.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    EvidenceReranker,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.fusion import reciprocal_rank_fusion
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan
from groundgraph.infrastructure.postgres.models import IndexVersion as _IndexVersionModel


class IndexVersionMismatchError(RuntimeError):
    """Raised when the active IndexVersion's embedding config does not match
    the configured EmbeddingProvider."""


@dataclass(frozen=True)
class RetrievalServiceConfig:
    """Configuration bundle for :class:`RetrievalService`."""

    session_factory: Any
    embedding_provider: EmbeddingProvider
    vector_retriever: VectorContentRetriever
    keyword_retriever: KeywordRetrieverPort
    reranker: EvidenceReranker
    answer_generator: AnswerGenerator
    settings: Settings | None = None


class RetrievalService:
    """Orchestrate hybrid vector + keyword retrieval, reranking, and answer generation.

    All cross-tenant access is blocked at the SQL layer (see
    :class:`PostgresVectorRetriever` / :class:`PostgresKeywordRetriever`).
    This service propagates the active IndexVersion and refuses to operate
    if the embedding configuration of the active index does not match the
    configured ``EmbeddingProvider`` (ADR-009).
    """

    def __init__(self, config: RetrievalServiceConfig) -> None:
        self._session_factory = config.session_factory
        self._embed = config.embedding_provider
        self._vector_retriever = config.vector_retriever
        self._keyword_retriever = config.keyword_retriever
        self._reranker = config.reranker
        self._answer_generator = config.answer_generator
        self._settings = config.settings or get_settings()

    async def _resolve_index_version(
        self,
        session: Any,
        index_name: str,
    ) -> _IndexVersionModel:
        """Resolve and validate the active IndexVersion for ``index_name``."""
        stmt = (
            select(_IndexVersionModel)
            .where(
                _IndexVersionModel.index_name == index_name,
                _IndexVersionModel.is_active == True,  # noqa: E712
            )
            .order_by(_IndexVersionModel.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        raw_idx = result.scalar_one_or_none()
        if raw_idx is None:
            raise RuntimeError(f"No active IndexVersion found for index_name={index_name!r}")
        idx = cast(_IndexVersionModel, raw_idx)
        if idx.embedding_model != self._embed.model:
            raise IndexVersionMismatchError(
                f"Active index {idx.version_id} uses embedding_model="
                f"{idx.embedding_model!r}, provider configured with "
                f"{self._embed.model!r}"
            )
        if idx.embedding_dimensions != self._embed.dimensions:
            raise IndexVersionMismatchError(
                f"Active index {idx.version_id} has embedding_dimensions="
                f"{idx.embedding_dimensions}, provider configured with "
                f"{self._embed.dimensions}"
            )
        return idx

    async def query(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        index_name: str | None = None,
        vector_top_k: int | None = None,
        final_limit: int | None = None,
    ) -> QueryResponse:
        if not tenant_id:
            raise ValueError("tenant_id is required for retrieval")

        index_name = index_name or self._settings.default_index_name
        vector_top_k = vector_top_k or self._settings.vector_top_k
        final_limit = final_limit or self._settings.final_evidence_limit

        async with self._session_factory() as session:
            active_index = await self._resolve_index_version(session, index_name)
            index_version_id = active_index.version_id

            query_vector = await self._embed.embed_one(question)

            vector_results = await self._vector_retriever.search(
                query_vector,
                vector_top_k,
                allowed_principals=[principal],
                tenant_id=tenant_id,
                index_version_id=index_version_id,
            )

            keyword_results = await self._keyword_retriever.search(
                question,
                self._settings.keyword_top_k,
                allowed_principals=[principal],
                tenant_id=tenant_id,
            )

        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        top_results = fused[:final_limit]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=r.source_id,
                document_id=r.document_id,
                version_id=r.version_id,
                chunk_id=r.chunk_id,
                content=r.content,
                retrieval_method="vector" if r.vector_score else "keyword",
                vector_score=r.vector_score,
                rerank_score=None,
                allowed_principals=list(r.allowed_principals),
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
