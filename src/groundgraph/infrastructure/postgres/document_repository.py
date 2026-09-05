"""PostgreSQL document repository implementation."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from groundgraph.application.ports import DocumentRepository
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.infrastructure.postgres.models import (
    Chunk as ChunkModel,
)
from groundgraph.infrastructure.postgres.models import (
    Document as DocumentModel,
)
from groundgraph.infrastructure.postgres.models import (
    DocumentVersion as DocumentVersionModel,
)
from groundgraph.infrastructure.postgres.models import (
    Source as SourceModel,
)
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def create_source(self, source: SourceDescriptor) -> SourceDescriptor:
        model = SourceModel(
            source_id=source.source_id,
            source_type=source.source_type,
            uri=source.uri,
            classification=source.classification,
            tenant_id=source.tenant_id,
            allowed_principals=source.allowed_principals,
        )
        self._session.add(model)
        await self._session.flush()
        return source

    async def get_source(self, source_id: UUID) -> SourceDescriptor | None:
        result = await self._session.execute(
            select(SourceModel).where(SourceModel.source_id == source_id)
        )
        row = result.scalar_one_or_none()
        return self._source_to_domain(row) if row else None

    async def list_sources(self) -> list[SourceDescriptor]:
        result = await self._session.execute(select(SourceModel).order_by(SourceModel.created_at))
        return [self._source_to_domain(row) for row in result.scalars().all()]

    async def create_document(self, document: ParsedDocument) -> ParsedDocument:
        doc_stmt = (
            pg_insert(DocumentModel)
            .values(
                document_id=document.document_id,
                source_id=document.source_id,
                title=document.title,
                media_type=document.media_type,
                current_version_id=document.version_id,
            )
            .on_conflict_do_nothing(index_elements=["document_id"])
        )
        version = DocumentVersionModel(
            version_id=document.version_id,
            document_id=document.document_id,
            checksum=document.checksum,
            content=document.content,
            doc_metadata=document.metadata,
            effective_at=document.effective_at,
            is_current=True,
        )
        self._session.add(doc_stmt)
        self._session.add(version)
        await self._session.flush()
        return document

    async def get_document(self, document_id: UUID) -> ParsedDocument | None:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.document_id == document_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        version = await self._get_current_version(row.document_id, row.current_version_id)
        if version is None:
            return None
        return self._document_version_to_domain(row, version)

    async def get_document_version(
        self, document_id: UUID, version_id: UUID
    ) -> ParsedDocument | None:
        doc_result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.document_id == document_id)
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            return None
        version_result = await self._session.execute(
            select(DocumentVersionModel).where(
                DocumentVersionModel.version_id == version_id,
                DocumentVersionModel.document_id == document_id,
            )
        )
        version = version_result.scalar_one_or_none()
        if version is None:
            return None
        return self._document_version_to_domain(document, version)

    async def list_document_versions(self, document_id: UUID) -> list[ParsedDocument]:
        doc_result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.document_id == document_id)
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            return []
        result = await self._session.execute(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .order_by(DocumentVersionModel.created_at)
        )
        return [
            self._document_version_to_domain(document, version)
            for version in result.scalars().all()
        ]

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        model = ChunkModel(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            ordinal=chunk.ordinal,
            heading_path=chunk.heading_path,
            content=chunk.content,
            token_count=chunk.token_count,
            checksum=chunk.checksum,
            start_locator=chunk.start_locator,
            end_locator=chunk.end_locator,
            allowed_principals=chunk.allowed_principals,
        )
        self._session.add(model)
        await self._session.flush()
        return chunk

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        result = await self._session.execute(
            select(ChunkModel).where(ChunkModel.chunk_id == chunk_id)
        )
        row = result.scalar_one_or_none()
        return self._chunk_to_domain(row) if row else None

    async def list_chunks(self, document_id: UUID, version_id: UUID) -> list[Chunk]:
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .where(ChunkModel.version_id == version_id)
            .order_by(ChunkModel.ordinal)
        )
        return [self._chunk_to_domain(row) for row in result.scalars().all()]

    async def delete_document(self, document_id: UUID) -> None:
        await self._session.execute(
            delete(DocumentModel).where(DocumentModel.document_id == document_id)
        )

    async def _get_current_version(
        self, document_id: UUID, current_version_id: UUID | None
    ) -> DocumentVersionModel | None:
        if current_version_id is not None:
            result = await self._session.execute(
                select(DocumentVersionModel).where(
                    DocumentVersionModel.version_id == current_version_id,
                    DocumentVersionModel.document_id == document_id,
                )
            )
            return result.scalar_one_or_none()
        result = await self._session.execute(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .where(DocumentVersionModel.is_current.is_(True))
            .order_by(DocumentVersionModel.created_at.desc())
        )
        return result.scalar_one_or_none()

    def _source_to_domain(self, row: SourceModel) -> SourceDescriptor:
        return SourceDescriptor(
            source_id=row.source_id,
            source_type=row.source_type,  # type: ignore[arg-type]
            uri=row.uri,
            classification=row.classification,
            tenant_id=row.tenant_id,
            allowed_principals=list(row.allowed_principals),
        )

    def _document_version_to_domain(
        self, document: DocumentModel, version: DocumentVersionModel
    ) -> ParsedDocument:
        return ParsedDocument(
            document_id=document.document_id,
            version_id=version.version_id,
            source_id=document.source_id,
            title=document.title,
            media_type=document.media_type,
            checksum=version.checksum,
            content=version.content,
            metadata=dict(version.doc_metadata),
            effective_at=version.effective_at,
        )

    def _chunk_to_domain(self, row: ChunkModel) -> Chunk:
        return Chunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            version_id=row.version_id,
            ordinal=row.ordinal,
            heading_path=list(row.heading_path),
            content=row.content,
            token_count=row.token_count,
            checksum=row.checksum,
            start_locator=row.start_locator,
            end_locator=row.end_locator,
            allowed_principals=list(row.allowed_principals),
        )
