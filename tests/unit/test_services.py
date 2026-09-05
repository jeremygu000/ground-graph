"""Unit tests for ingestion service."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import Chunk, ParsedDocument


class _FakeObjectStore:
    def __init__(self) -> None:
        self.put_raw_calls: list[tuple[str, bytes, str | None]] = []

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.put_raw_calls.append((key, data, content_type))


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[ParsedDocument] = []
        self.chunks: list[Chunk] = []

    async def create_document(self, doc: ParsedDocument) -> None:
        self.documents.append(doc)

    async def create_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)


class TestIngestionService:
    def setup_method(self) -> None:
        self.fake_docs: Any = _FakeDocumentRepository()
        self.fake_store: Any = _FakeObjectStore()
        self.service = IngestionService(
            documents=self.fake_docs,
            object_store=self.fake_store,
            chunker=Chunker(),
        )

    async def test_ingest_file_parses_and_stores(self, tmp_path: Any) -> None:
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello world\nThis is a test.")

        doc_id, ver_id = await self.service.ingest_file(
            source_id=uuid4(),
            file_path=str(file_path),
            media_type="text/plain",
            allowed_principals=["engineering"],
            tenant_id="tenant-a",
        )

        assert doc_id is not None
        assert ver_id is not None
        assert len(self.fake_docs.documents) == 1
        assert self.fake_docs.documents[0].title == "Hello world"
        assert len(self.fake_store.put_raw_calls) == 1

    async def test_ingest_file_stores_chunks(self, tmp_path: Any) -> None:
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"word " * 500)

        await self.service.ingest_file(
            source_id=uuid4(),
            file_path=str(file_path),
            media_type="text/plain",
            allowed_principals=["engineering"],
            tenant_id="tenant-a",
        )

        assert len(self.fake_docs.chunks) > 0
        for chunk in self.fake_docs.chunks:
            assert chunk.document_id is not None
            assert chunk.version_id is not None
            assert chunk.allowed_principals == ["engineering"]

    async def test_ingest_file_unsupported_media_type_raises(self, tmp_path: Any) -> None:
        file_path = tmp_path / "doc.xyz"
        file_path.write_bytes(b"data")

        with pytest.raises(ValueError, match="Unsupported media type"):
            await self.service.ingest_file(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="application/xyz",
                allowed_principals=["engineering"],
                tenant_id="tenant-a",
            )
