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
        allowed_principals: list[str],
        tenant_id: str,
    ) -> tuple[UUID, UUID]:
        raw_bytes = await self._read_file(file_path)
        checksum = self._sha256(raw_bytes)

        document_id = uuid4()
        version_id = uuid4()

        raw_key = f"sources/{source_id}/{document_id}/v1/raw"
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
            allowed_principals=allowed_principals,
        )
        for chunk in chunks:
            await self._documents.create_chunk(chunk)

        return document_id, version_id

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
