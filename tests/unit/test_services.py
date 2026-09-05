"""Unit tests for ingestion service."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.domain.evidence import OutboxEvent


class _FakeObjectStore:
    def __init__(self) -> None:
        self.put_raw_calls: list[tuple[str, bytes, str | None]] = []

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.put_raw_calls.append((key, data, content_type))


class _FakeOutboxRepository:
    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    async def add(self, event: OutboxEvent) -> OutboxEvent:
        self.events.append(event)
        return event


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
        self.fake_outbox: Any = _FakeOutboxRepository()
        self.service = IngestionService(
            documents=self.fake_docs,
            object_store=self.fake_store,
            outbox_repo=self.fake_outbox,
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

        result = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/markdown",
        )

        assert result.document_id is not None
        assert result.version_id is not None
        assert result.created_new_version is True
        assert result.tenant_id == "tenant-a"
        assert len(self.fake_docs.documents) == 1
        assert self.fake_docs.documents[0].title == "Hello"
        assert len(self.fake_store.put_raw_calls) == 1

    async def test_ingest_file_emits_outbox_event(self, tmp_path: Any) -> None:
        source_id = uuid4()
        self.fake_docs.set_source(
            SourceDescriptor(
                source_id=source_id,
                source_type="filesystem",
                uri=str(tmp_path),
                classification="internal",
                tenant_id="tenant-b",
                allowed_principals=["eng"],
            )
        )
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Content")

        result = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert len(self.fake_outbox.events) == 1
        event = self.fake_outbox.events[0]
        assert event.payload["tenant_id"] == "tenant-b"
        assert event.payload["document_id"] == str(result.document_id)

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

    async def test_ingest_file_idempotent_same_checksum_no_outbox(self, tmp_path: Any) -> None:
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

        result1 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )
        assert result1.created_new_version is True
        assert len(self.fake_outbox.events) == 1

        self.fake_docs.set_find_result((result1.document_id, result1.version_id))

        result2 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert result2.document_id == result1.document_id
        assert result2.version_id == result1.version_id
        assert result2.created_new_version is False
        assert len(self.fake_outbox.events) == 1

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

        result1 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        self.fake_docs.set_find_result((result1.document_id, result1.version_id))
        self.fake_docs.set_checksum_override(result1.document_id, result1.version_id, "v1_checksum")

        file_path.write_bytes(b"Version 2 content, different!")

        result2 = await self.service.ingest_file(
            source_id=source_id,
            file_path=str(file_path),
            media_type="text/plain",
        )

        assert result2.document_id == result1.document_id
        assert result2.version_id != result1.version_id
        assert result2.created_new_version is True
        assert len(self.fake_docs.documents) == 2
        assert len(self.fake_outbox.events) == 2

    async def test_ingest_file_path_escape_rejected(self, tmp_path: Any) -> None:
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
        evil = tmp_path.parent / "evil.txt"
        evil.write_bytes(b"secret")

        with pytest.raises(ValueError, match="outside source root"):
            await self.service.ingest_file(
                source_id=source_id,
                file_path=str(evil),
                media_type="text/plain",
            )

    async def test_ingest_file_symlink_rejected(self, tmp_path: Any) -> None:
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
        target = tmp_path / "target.txt"
        target.write_bytes(b"secret")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="symlink not allowed"):
            await self.service.ingest_file(
                source_id=source_id,
                file_path=str(link),
                media_type="text/plain",
            )
