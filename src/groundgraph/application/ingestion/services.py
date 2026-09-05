"""Ingestion service: orchestrates file acquisition, parsing, chunking, and storage."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from groundgraph.application.ports import DocumentRepository, ObjectStore, OutboxRepository
from groundgraph.domain.documents import ParsedDocument
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType

from .chunker import Chunker
from .parsers import ParsedContent, ParserRegistry


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    version_id: UUID
    created_new_version: bool
    tenant_id: str


class IngestionService:
    def __init__(
        self,
        documents: DocumentRepository,
        object_store: ObjectStore,
        outbox_repo: OutboxRepository,
        chunker: Chunker | None = None,
    ) -> None:
        self._documents = documents
        self._object_store = object_store
        self._outbox = outbox_repo
        self._chunker = chunker or Chunker()

    async def ingest_file(
        self,
        source_id: UUID,
        file_path: str,
        media_type: str,
    ) -> IngestionResult:
        source = await self._documents.get_source(source_id)
        if source is None:
            raise ValueError(f"source not found: {source_id}")

        resolved_path = self._resolve_safe_path(source.uri, file_path)
        self._check_size(resolved_path)
        raw_bytes = await self._read_file(resolved_path)

        checksum = self._sha256(raw_bytes)

        existing = await self._documents.find_active_document_by_source(source_id, file_path)
        if existing is not None:
            doc_id, ver_id = existing
            current = await self._documents.get_document_version(doc_id, ver_id)
            if current is not None and current.checksum == checksum:
                return IngestionResult(
                    document_id=doc_id,
                    version_id=ver_id,
                    created_new_version=False,
                    tenant_id=source.tenant_id,
                )
            document_id, version_id = doc_id, uuid4()
        else:
            document_id, version_id = uuid4(), uuid4()

        raw_key = f"sources/{source_id}/{document_id}/{version_id}/raw"
        await self._object_store.put_raw(raw_key, raw_bytes, media_type)

        parsed = self._parse(raw_bytes, media_type)

        document = ParsedDocument(
            document_id=document_id,
            version_id=version_id,
            source_id=source_id,
            title=parsed.title,
            media_type=media_type,
            checksum=checksum,
            content=parsed.body,
            metadata={
                "file_path": file_path,
                **parsed.metadata,
            },
            effective_at=datetime.now(UTC),
        )
        await self._documents.create_document(document)

        chunks = self._chunker.chunk(
            content=parsed,
            document_id=document_id,
            version_id=version_id,
            allowed_principals=source.allowed_principals,
        )
        for chunk in chunks:
            await self._documents.create_chunk(chunk)

        event = OutboxEvent(
            event_id=uuid4(),
            aggregate_type="document",
            aggregate_id=document_id,
            event_type=OutboxEventType.DOCUMENT_PARSED,
            payload={
                "document_id": str(document_id),
                "version_id": str(version_id),
                "source_id": str(source_id),
                "tenant_id": source.tenant_id,
            },
            created_at=datetime.now(UTC),
        )
        await self._outbox.add(event)

        return IngestionResult(
            document_id=document_id,
            version_id=version_id,
            created_new_version=True,
            tenant_id=source.tenant_id,
        )

    MAX_FILE_SIZE = 100 * 1024 * 1024

    def _resolve_safe_path(self, source_root: str, file_path: str) -> Path:
        root = Path(source_root).resolve()
        requested = Path(file_path)
        if requested.is_symlink():
            raise ValueError(f"symlink not allowed: {file_path}")
        requested_resolved = requested.resolve()
        try:
            requested_resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"file path outside source root: {file_path}") from exc
        return requested_resolved

    def _check_size(self, path: Path) -> None:
        size = path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            raise ValueError(f"file exceeds maximum size: {size} > {self.MAX_FILE_SIZE}")

    async def _read_file(self, path: Path) -> bytes:
        return await self._read_async(path)

    async def _read_async(self, path: Path) -> bytes:
        def _read() -> bytes:
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    def _parse(self, content: bytes, media_type: str) -> ParsedContent:
        parser = ParserRegistry.get(media_type)
        if parser is None:
            raise ValueError(f"Unsupported media type: {media_type}")
        return parser.parse(content)

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
