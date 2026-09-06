"""Unit tests for ingestion service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import (
    Chunk,
    IngestionCheckpoint,
    IngestionCheckpointStatus,
    ParsedDocument,
    SourceDescriptor,
)
from groundgraph.domain.evidence import OutboxEvent


class _FakeObjectStore:
    def __init__(self) -> None:
        self.put_raw_calls: list[tuple[str, bytes, str | None]] = []
        self._objects: dict[str, bytes] = {}

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.put_raw_calls.append((key, data, content_type))
        self._objects[key] = data

    async def exists(self, key: str) -> bool:
        return key in self._objects


class _FakeOutboxRepository:
    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    async def add(self, event: OutboxEvent) -> OutboxEvent:
        self.events.append(event)
        return event


class _FakeIngestionCheckpointRepository:
    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str, str], IngestionCheckpoint] = {}

    async def upsert_checkpoint(  # noqa: PLR0917
        self,
        source_id: Any,
        canonical_locator: str,
        content_checksum: str,
        status: IngestionCheckpointStatus,
        document_id: Any | None = None,
        version_id: Any | None = None,
        error_message: str | None = None,
    ) -> IngestionCheckpoint:
        now = datetime.now(UTC)
        key = (str(source_id), canonical_locator, content_checksum)
        self._checkpoints[key] = IngestionCheckpoint(
            checkpoint_id=uuid4(),
            source_id=source_id,
            canonical_locator=canonical_locator,
            content_checksum=content_checksum,
            status=status,
            document_id=document_id,
            version_id=version_id,
            error_message=error_message,
            created_at=now,
            updated_at=now,
            completed_at=now
            if status in (IngestionCheckpointStatus.PERSISTED, IngestionCheckpointStatus.FAILED)
            else None,
        )
        return self._checkpoints[key]

    async def get_checkpoint(
        self, source_id: Any, canonical_locator: str, content_checksum: str
    ) -> IngestionCheckpoint | None:
        return self._checkpoints.get((str(source_id), canonical_locator, content_checksum))


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[ParsedDocument] = []
        self.chunks: list[Chunk] = []
        self.sources: dict[str, SourceDescriptor] = {}
        self._find_result: tuple[Any, Any] | None = None
        self._checksum_override: dict[tuple[Any, Any], str] = {}
        self._versions: dict[tuple[Any, Any], ParsedDocument] = {}

    def set_source(self, source: SourceDescriptor) -> None:
        self.sources[str(source.source_id)] = source

    def set_find_result(self, result: tuple[Any, Any] | None) -> None:
        self._find_result = result

    def set_checksum_override(self, document_id: Any, version_id: Any, checksum: str) -> None:
        self._checksum_override[(document_id, version_id)] = checksum

    async def find_or_create_source(self, source: SourceDescriptor) -> SourceDescriptor:
        for s in self.sources.values():
            if (
                s.tenant_id == source.tenant_id
                and s.source_type == source.source_type
                and s.uri == source.uri
            ):
                return s
        self.sources[str(source.source_id)] = source
        return source

    async def get_source(self, source_id: Any) -> SourceDescriptor | None:
        return self.sources.get(str(source_id))

    async def find_active_document_by_canonical_locator(
        self, source_id: Any, canonical_locator: str
    ) -> tuple[Any, Any] | None:
        return self._find_result

    async def get_document_version(
        self, document_id: Any, version_id: Any
    ) -> ParsedDocument | None:
        key = (document_id, version_id)
        if key in self._versions:
            doc = self._versions[key]
            override = self._checksum_override.get(key)
            if override:
                return ParsedDocument(**{**doc.model_dump(), "checksum": override})
            return doc
        for doc in self.documents:
            if doc.document_id == document_id and doc.version_id == version_id:
                self._versions[key] = doc
                override = self._checksum_override.get(key)
                if override:
                    return ParsedDocument(**{**doc.model_dump(), "checksum": override})
                return doc
        return None

    async def upsert_document(self, doc: ParsedDocument) -> tuple[ParsedDocument, bool]:
        for i, existing in enumerate(self.documents):
            if (
                existing.source_id == doc.source_id
                and existing.source_locator == doc.source_locator
            ):
                self.documents[i] = doc
                self._versions[(doc.document_id, doc.version_id)] = doc
                return (doc, False)
        self.documents.append(doc)
        self._versions[(doc.document_id, doc.version_id)] = doc
        return (doc, True)

    async def create_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    async def get_document(self, document_id: Any) -> ParsedDocument | None:
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None


class _FakeIngestionUoW:
    def __init__(
        self,
        docs: _FakeDocumentRepository,
        outbox: _FakeOutboxRepository,
        checkpoint: _FakeIngestionCheckpointRepository,
    ) -> None:
        self.documents = docs
        self.outbox = outbox
        self.ingestion_checkpoint = checkpoint
        self._committed = False

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: object) -> None:
        self._committed = True

    async def commit(self) -> None:
        self._committed = True

    async def rollback(self) -> None:
        pass


class TestIngestionService:
    def setup_method(self) -> None:
        self.fake_docs = _FakeDocumentRepository()
        self.fake_store: Any = _FakeObjectStore()
        self.fake_outbox = _FakeOutboxRepository()
        self.fake_checkpoint = _FakeIngestionCheckpointRepository()

        def uow_factory() -> Any:
            return _FakeIngestionUoW(self.fake_docs, self.fake_outbox, self.fake_checkpoint)

        self.service = IngestionService(
            uow_factory=uow_factory,
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

    async def test_ingest_file_idempotent_same_checksum_no_new_version(self, tmp_path: Any) -> None:
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
        assert len(self.fake_docs.documents) == 1
        assert len(self.fake_docs._versions) == 2
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
