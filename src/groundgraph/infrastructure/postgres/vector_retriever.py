"""pgvector similarity search with pre-retrieval ACL filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select

from groundgraph.application.ports import VectorRetriever
from groundgraph.infrastructure.postgres.models import Chunk as ChunkModel
from groundgraph.infrastructure.postgres.models import ChunkEmbedding as ChunkEmbeddingModel
from groundgraph.infrastructure.postgres.models import Document as DocumentModel
from groundgraph.infrastructure.postgres.models import IndexVersion as IndexVersionModel
from groundgraph.infrastructure.postgres.models import Source as SourceModel
from groundgraph.infrastructure.postgres.session import PostgresSession


@dataclass
class VectorSearchResult:
    chunk_id: UUID
    source_id: UUID
    document_id: UUID
    version_id: UUID
    content: str
    score: float
    allowed_principals: list[str]


class PostgresVectorRetriever(VectorRetriever):
    """pgvector similarity search that filters by ACL *before* returning results.

    The retrieval SQL joins chunk → source and filters on
    ``source.is_active`` and ``chunk_id IN (authorized_chunks)``.
    Only active index versions are used for retrieval.
    """

    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[UUID, float]]:
        """Return (chunk_id, cosine_distance) pairs, sorted ascending."""
        allowed_principals: list[str] | None = None
        source_ids: list[UUID] | None = None
        index_version_id: UUID | None = None
        tenant_id: str | None = None

        if filters:
            allowed_principals = filters.get("allowed_principals")
            source_ids = filters.get("source_ids")
            index_version_id = filters.get("index_version_id")
            tenant_id = filters.get("tenant_id")

        if not tenant_id:
            raise ValueError("tenant_id is required for retrieval")

        active_index = await self._get_active_index_version(index_version_id)
        if active_index is None:
            return []

        conditions = [
            ChunkEmbeddingModel.index_version_id == active_index.version_id,
            SourceModel.is_active == True,  # noqa: E712
            SourceModel.tenant_id == tenant_id,
        ]

        if allowed_principals is not None:
            conditions.append(SourceModel.allowed_principals.overlap(allowed_principals))

        if source_ids is not None:
            conditions.append(DocumentModel.source_id.in_(source_ids))

        stmt = (
            select(
                ChunkModel.chunk_id,
                ChunkEmbeddingModel.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(
                ChunkEmbeddingModel,
                ChunkModel.chunk_id == ChunkEmbeddingModel.chunk_id,
            )
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.document_id)
            .join(SourceModel, DocumentModel.source_id == SourceModel.source_id)
            .where(*conditions)
            .order_by(ChunkEmbeddingModel.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )

        result = await self._session.execute(stmt)
        return [(row.chunk_id, row.distance) for row in result.all()]

    async def search_with_content(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return full search results, ACL + tenant filtered."""
        allowed_principals: list[str] | None = None
        source_ids: list[UUID] | None = None
        index_version_id: UUID | None = None
        tenant_id: str | None = None

        if filters:
            allowed_principals = filters.get("allowed_principals")
            source_ids = filters.get("source_ids")
            index_version_id = filters.get("index_version_id")
            tenant_id = filters.get("tenant_id")

        if not tenant_id:
            raise ValueError("tenant_id is required for retrieval")

        active_index = await self._get_active_index_version(index_version_id)
        if active_index is None:
            return []

        conditions = [
            ChunkEmbeddingModel.index_version_id == active_index.version_id,
            SourceModel.is_active == True,  # noqa: E712
            SourceModel.tenant_id == tenant_id,
        ]

        if allowed_principals is not None:
            conditions.append(SourceModel.allowed_principals.overlap(allowed_principals))

        if source_ids is not None:
            conditions.append(DocumentModel.source_id.in_(source_ids))

        stmt = (
            select(
                ChunkModel.chunk_id,
                DocumentModel.source_id,
                ChunkModel.document_id,
                ChunkModel.version_id,
                ChunkModel.content,
                ChunkEmbeddingModel.embedding.cosine_distance(query_vector).label("distance"),
                SourceModel.allowed_principals,
            )
            .join(
                ChunkEmbeddingModel,
                ChunkModel.chunk_id == ChunkEmbeddingModel.chunk_id,
            )
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.document_id)
            .join(SourceModel, DocumentModel.source_id == SourceModel.source_id)
            .where(*conditions)
            .order_by(ChunkEmbeddingModel.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )

        result = await self._session.execute(stmt)
        return [
            VectorSearchResult(
                chunk_id=row.chunk_id,
                source_id=row.source_id,
                document_id=row.document_id,
                version_id=row.version_id,
                content=row.content,
                score=row.distance,
                allowed_principals=list(row.allowed_principals) if row.allowed_principals else [],
            )
            for row in result.all()
        ]

    async def _get_active_index_version(
        self, index_version_id: UUID | None = None
    ) -> IndexVersionModel | None:
        if index_version_id:
            stmt = select(IndexVersionModel).where(
                IndexVersionModel.version_id == index_version_id,
                IndexVersionModel.is_active == True,  # noqa: E712
            )
        else:
            stmt = (
                select(IndexVersionModel)
                .where(IndexVersionModel.is_active == True)  # noqa: E712
                .order_by(IndexVersionModel.created_at.desc())
                .limit(1)
            )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
