"""Ingestion service: orchestrates file acquisition, parsing, chunking, and storage."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from groundgraph.application.ports import DocumentRepository, ObjectStore
from groundgraph.domain.documents import ParsedDocument

from .chunker import Chunker
from .parsers import ParsedContent, ParserRegistry


class IngestionService:
    def __init__(
        self,
        documents: DocumentRepository,
        object_store: ObjectStore,
        chunker: Chunker | None = None,
    ) -> None:
        self._documents = documents
        self._object_store = object_store
        self._chunker = chunker or Chunker()

    async def ingest_file(
        self,
        source_id: UUID,
        file_path: str,
        media_type: str,
    ) -> tuple[UUID, UUID]:
        source = await self._documents.get_source(source_id)
        if source is None:
            raise ValueError(f"source not found: {source_id}")

        resolved_path = self._resolve_safe_path(source.uri, file_path)

        raw_bytes = await self._read_file(str(resolved_path))
        self._check_size(raw_bytes)

        checksum = self._sha256(raw_bytes)

        existing = await self._documents.find_active_document_by_source(source_id, file_path)
        if existing is not None:
            doc_id, ver_id = existing
            current = await self._documents.get_document_version(doc_id, ver_id)
            if current is not None and current.checksum == checksum:
                return (doc_id, ver_id)
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

        return document_id, version_id

    MAX_FILE_SIZE = 100 * 1024 * 1024

    def _resolve_safe_path(self, source_root: str, file_path: str) -> Path:
        root = Path(source_root).resolve()
        requested = Path(file_path).resolve()
        if not str(requested).startswith(str(root)):
            raise ValueError(f"file path outside source root: {file_path}")
        if requested.is_symlink():
            raise ValueError(f"symlink escape not allowed: {file_path}")
        return requested

    def _check_size(self, data: bytes) -> None:
        if len(data) > self.MAX_FILE_SIZE:
            raise ValueError(f"file exceeds maximum size: {len(data)} > {self.MAX_FILE_SIZE}")

    async def _read_file(self, path: str) -> bytes:
        p = Path(path)
        return await self._read_async(p)

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
