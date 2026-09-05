"""Minimal PostgreSQL document repository adapter.

This file exists to make the UoW composition concrete while the full
document persistence adapter is still being finalized.
"""

from __future__ import annotations

from uuid import UUID

from groundgraph.application.ports import DocumentRepository
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresDocumentRepository(DocumentRepository):
    """DocumentRepository adapter placeholder backed by the current session.

    This implements the port surface used by M2 UoW wiring. The real
    ingestion/document CRUD behavior will be expanded in the next slice,
    but the methods are present and typed now.
    """

    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def create_source(self, source: SourceDescriptor) -> SourceDescriptor:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def get_source(self, source_id: UUID) -> SourceDescriptor | None:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def list_sources(self) -> list[SourceDescriptor]:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def create_document(self, document: ParsedDocument) -> ParsedDocument:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def get_document(self, document_id: UUID) -> ParsedDocument | None:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def get_document_version(
        self, document_id: UUID, version_id: UUID
    ) -> ParsedDocument | None:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def list_document_versions(self, document_id: UUID) -> list[ParsedDocument]:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        raise NotImplementedError("document repository persistence is not wired yet")

    async def list_chunks(self, document_id: UUID, version_id: UUID) -> list[Chunk]:
        raise NotImplementedError("document repository persistence is not wired yet")
