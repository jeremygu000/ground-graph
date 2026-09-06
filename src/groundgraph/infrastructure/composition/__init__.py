"""Infrastructure adapters that implement the application retrieval ports.

These are the *only* place where the application ``RetrievedChunk`` value
object is constructed from infrastructure rows; the application layer
itself never imports from ``groundgraph.infrastructure``.
"""

from __future__ import annotations

from uuid import UUID

from groundgraph.application.ports import (
    KeywordRetrieverPort,
    RetrievedChunk,
    VectorContentRetriever,
)
from groundgraph.infrastructure.postgres.keyword_retriever import (
    KeywordSearchResult,
    PostgresKeywordRetriever,
)
from groundgraph.infrastructure.postgres.vector_retriever import (
    PostgresVectorRetriever,
    VectorSearchResult,
)


class PgVectorContentRetriever(VectorContentRetriever):
    """Adapter that implements :class:`VectorContentRetriever` against
    :class:`PostgresVectorRetriever`, converting ORM rows into the
    application-layer :class:`RetrievedChunk`.
    """

    def __init__(self, retriever: PostgresVectorRetriever) -> None:
        self._retriever = retriever

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        allowed_principals: list[str],
        tenant_id: str,
        index_version_id: UUID,
    ) -> list[RetrievedChunk]:
        results: list[VectorSearchResult] = await self._retriever.search_with_content(
            query_vector,
            top_k=top_k,
            filters={
                "allowed_principals": allowed_principals,
                "tenant_id": tenant_id,
                "index_version_id": index_version_id,
            },
        )
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                source_id=r.source_id,
                document_id=r.document_id,
                version_id=r.version_id,
                content=r.content,
                vector_score=r.score,
                keyword_score=None,
                allowed_principals=list(r.allowed_principals),
            )
            for r in results
        ]


class PgKeywordRetrieverAdapter(KeywordRetrieverPort):
    """Adapter that implements :class:`KeywordRetrieverPort` against
    :class:`PostgresKeywordRetriever`.
    """

    def __init__(self, retriever: PostgresKeywordRetriever) -> None:
        self._retriever = retriever

    async def search(
        self,
        query: str,
        top_k: int,
        *,
        allowed_principals: list[str],
        tenant_id: str,
    ) -> list[RetrievedChunk]:
        results: list[KeywordSearchResult] = await self._retriever.search(
            query,
            top_k=top_k,
            allowed_principals=allowed_principals,
            tenant_id=tenant_id,
        )
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                source_id=r.source_id,
                document_id=r.document_id,
                version_id=r.version_id,
                content=r.content,
                vector_score=None,
                keyword_score=r.rank,
                allowed_principals=list(r.allowed_principals),
            )
            for r in results
        ]
