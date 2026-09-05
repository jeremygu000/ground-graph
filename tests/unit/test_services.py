"""Unit tests for ingestion service."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor


class _FakeObjectStore:
    def __init__(self) -> None:
        self.put_raw_calls: list[tuple[str, bytes, str | None]] = []

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.put_raw_calls.append((key, data, content_type))


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[ParsedDocument] = []
        self.chunks: list[Chunk] = []
        self.sources: dict[str, SourceDescriptor] = {}
        self._find_result: tuple[Any, Any] | None = None
        self._checksum_override: dict[tuple[Any, Any], str] = {}

    def set_source(self, source: SourceDescriptor) -> None:
        self.sources[str(source.source_id)] = source

    def set_find_result(self, result: tuple[Any, Any] | None) -> None:
        self._find_result = result

    def set_checksum_override(self, document_id: Any, version_id: Any, checksum: str) -> None:
        self._checksum_override[(document_id, version_id)] = checksum

    async def get_source(self, source_id: Any) -> SourceDescriptor | None:
        return self.sources.get(str(source_id))

    async def find_active_document_by_source(
        self, source_id: Any, file_path: str
    ) -> tuple[Any, Any] | None:
        return self._find_result

    async def get_document_version(
        self, document_id: Any, version_id: Any
    ) -> ParsedDocument | None:
        for doc in self.documents:
            if doc.document_id == document_id and doc.version_id == version_id:
                override = self._checksum_override.get((document_id, version_id))
                if override:
                    return ParsedDocument(**{**doc.model_dump(), "checksum": override})
                return doc
        return None

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
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"# Hello\n\nWorld test.")

        doc_id, ver_id = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/markdown",
        )

        assert doc_id is not None
        assert ver_id is not None
        assert len(self.fake_docs.documents) == 1
        assert self.fake_docs.documents[0].title == "Hello"
        assert len(self.fake_store.put_raw_calls) == 1

    async def test_ingest_file_stores_chunks(self, tmp_path: Any) -> None:
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"word " * 500)

        await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert len(self.fake_docs.chunks) > 0
        for chunk in self.fake_docs.chunks:
            assert chunk.document_id is not None
            assert chunk.version_id is not None
            assert chunk.allowed_principals == ["engineering"]

    async def test_ingest_file_unsupported_media_type_raises(self, tmp_path: Any) -> None:
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        file_path = tmp_path / "doc.xyz"
        file_path.write_bytes(b"data")

        with pytest.raises(ValueError, match="Unsupported media type"):
            await self.service.ingest_file(
                source_id=source_id,
                file_path=str(file_path),
                media_type="application/xyz",
            )

    async def test_ingest_file_unknown_source_raises(self, tmp_path: Any) -> None:
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello")

        with pytest.raises(ValueError, match="source not found"):
            await self.service.ingest_file(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
            )

    async def test_ingest_file_idempotent_same_checksum(self, tmp_path: Any) -> None:
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello world")

        doc1, ver1 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        self.fake_docs.set_find_result((doc1, ver1))

        doc2, ver2 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert doc2 == doc1
        assert ver2 == ver1
        assert len(self.fake_docs.documents) == 1

    async def test_ingest_file_new_version_on_changed_content(self, tmp_path: Any) -> None:
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Version 1 content")

        doc1, ver1 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        self.fake_docs.set_find_result((doc1, ver1))
        self.fake_docs.set_checksum_override(doc1, ver1, "v1_checksum")
        self.fake_docs.documents[0].metadata["file_path"] = str(file_path)

        file_path.write_bytes(b"Version 2 content, different!")

        doc2, ver2 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert doc2 == doc1
        assert ver2 != ver1
        assert len(self.fake_docs.documents) == 2
