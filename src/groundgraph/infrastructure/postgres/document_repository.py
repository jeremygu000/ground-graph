"""PostgreSQL document repository implementation."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from groundgraph.application.ports import DocumentRepository
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.infrastructure.postgres.json_utils import snapshot_json_object
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

    async def find_or_create_source(self, source: SourceDescriptor) -> SourceDescriptor:
        stmt = (
            pg_insert(SourceModel)
            .values(
                source_id=source.source_id,
                source_type=source.source_type,
                uri=source.uri,
                classification=source.classification,
                tenant_id=source.tenant_id,
                allowed_principals=source.allowed_principals,
            )
            .on_conflict_do_nothing(index_elements=["tenant_id", "source_type", "uri"])
            .returning(SourceModel)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            return self._source_to_domain(row)
        result = await self._session.execute(
            select(SourceModel).where(
                SourceModel.tenant_id == source.tenant_id,
                SourceModel.source_type == source.source_type,
                SourceModel.uri == source.uri,
            )
        )
        existing = result.scalar_one()
        return self._source_to_domain(existing)

    async def get_source(self, source_id: UUID) -> SourceDescriptor | None:
        result = await self._session.execute(
            select(SourceModel).where(SourceModel.source_id == source_id)
        )
        row = result.scalar_one_or_none()
        return self._source_to_domain(row) if row else None

    async def list_sources(self) -> list[SourceDescriptor]:
        result = await self._session.execute(select(SourceModel).order_by(SourceModel.created_at))
        return [self._source_to_domain(row) for row in result.scalars().all()]

    async def find_active_document_by_canonical_locator(
        self, source_id: UUID, canonical_locator: str
    ) -> tuple[UUID, UUID] | None:
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.source_id == source_id,
                DocumentModel.source_locator == canonical_locator,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None or doc.current_version_id is None:
            return None
        return (doc.document_id, doc.current_version_id)

    async def upsert_document(self, document: ParsedDocument) -> tuple[ParsedDocument, bool]:
        stmt = (
            pg_insert(DocumentModel)
            .values(
                document_id=document.document_id,
                source_id=document.source_id,
                source_locator=document.source_locator,
                title=document.title,
                media_type=document.media_type,
                current_version_id=document.version_id,
            )
            .on_conflict_do_nothing(index_elements=["source_id", "source_locator"])
            .returning(DocumentModel.document_id)
        )
        result = await self._session.execute(stmt)
        winner_id_row = result.scalar_one_or_none()

        is_new = winner_id_row is not None
        document_id: UUID
        if is_new:
            document_id = document.document_id
        else:
            winner_result = await self._session.execute(
                select(DocumentModel.document_id).where(
                    DocumentModel.source_id == document.source_id,
                    DocumentModel.source_locator == document.source_locator,
                )
            )
            document_id = winner_result.scalar_one()

        if not is_new:
            await self._session.execute(
                update(DocumentModel)
                .where(
                    DocumentModel.document_id == document_id,
                )
                .values(
                    current_version_id=document.version_id,
                    title=document.title,
                    media_type=document.media_type,
                )
            )
            await self._session.execute(
                update(DocumentVersionModel)
                .where(
                    DocumentVersionModel.document_id == document_id,
                    DocumentVersionModel.version_id != document.version_id,
                )
                .values(is_current=False)
            )

        version = DocumentVersionModel(
            version_id=document.version_id,
            document_id=document_id,
            checksum=document.checksum,
            content=document.content,
            doc_metadata=snapshot_json_object(document.metadata),
            effective_at=document.effective_at,
            is_current=True,
        )
        self._session.add(version)
        await self._session.flush()

        canonical = ParsedDocument(
            document_id=document_id,
            version_id=document.version_id,
            source_id=document.source_id,
            source_locator=document.source_locator,
            title=document.title,
            media_type=document.media_type,
            checksum=document.checksum,
            content=document.content,
            metadata=document.metadata,
            effective_at=document.effective_at,
        )
        return (canonical, is_new)

    async def create_document(self, document: ParsedDocument) -> ParsedDocument:
        doc, _ = await self.upsert_document(document)
        return doc

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
            update(DocumentModel)
            .where(DocumentModel.document_id == document_id)
            .values(current_version_id=None)
        )
        await self._session.execute(
            update(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .values(is_current=False)
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
            source_locator=document.source_locator,
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
