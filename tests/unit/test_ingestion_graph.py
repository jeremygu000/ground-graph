"""Unit tests for the LangGraph ingestion workflow.

Coverage target: src/groundgraph/workflows/ingestion_graph.py
The workflow composes acquire → parse → chunk → store → emit-outbox via
application services; this test exercises each node and the orchestrating
graph with in-memory fakes.
"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import uuid4

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.domain.documents import Chunk, ParsedDocument
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType
from groundgraph.workflows.ingestion_graph import IngestionState, IngestionWorkflow


class _FakeObjectStore:
    def __init__(self) -> None:
        self.put_raw_calls: list[tuple[str, bytes, str | None]] = []

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        self.put_raw_calls.append((key, data, content_type))


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.documents: list[ParsedDocument] = []
        self.chunks: list[Chunk] = []

    async def create_document(self, document: ParsedDocument) -> None:
        self.documents.append(document)

    async def create_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)


class _FakeOutboxRepository:
    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    async def add(self, event: OutboxEvent) -> OutboxEvent:
        self.events.append(event)
        return event


def _build_workflow() -> tuple[
    IngestionWorkflow, _FakeObjectStore, _FakeDocumentRepository, _FakeOutboxRepository
]:
    docs: Any = _FakeDocumentRepository()
    outbox: Any = _FakeOutboxRepository()
    store: Any = _FakeObjectStore()
    workflow = IngestionWorkflow(
        documents=docs,
        outbox_repo=outbox,
        object_store=store,
    )
    return workflow, store, docs, outbox


class TestIngestionWorkflowHappyPath:
    async def test_runs_full_pipeline_end_to_end(self, tmp_path: Any) -> None:
        workflow, store, docs, outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello world\nSecond line here.")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )

        assert result.error is None
        assert result.document_id is not None
        assert result.version_id is not None
        assert result.outbox_event_id is not None
        assert result.raw_bytes == b"Hello world\nSecond line here."
        assert result.telemetry["acquire_bytes"] == len(b"Hello world\nSecond line here.")
        assert "parse_tokens" in result.telemetry
        assert result.telemetry["chunks_created"] == result.chunks_created

        assert len(store.put_raw_calls) == 1
        assert len(docs.documents) == 1
        assert len(docs.chunks) == result.chunks_created
        assert len(outbox.events) == 1

    async def test_stored_document_has_expected_metadata(self, tmp_path: Any) -> None:
        workflow, _store, docs, _outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"First line becomes title\nBody content here.")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )

        stored = docs.documents[0]
        assert stored.title == "First line becomes title"
        assert stored.media_type == "text/plain"
        assert (
            stored.checksum
            == hashlib.sha256(b"First line becomes title\nBody content here.").hexdigest()
        )
        assert stored.metadata.get("file_path") == str(file_path)
        assert stored.source_id == result.source_id

    async def test_outbox_event_contains_required_ids(self, tmp_path: Any) -> None:
        workflow, _store, _docs, outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Title\nBody")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )

        event = outbox.events[0]
        assert isinstance(event, OutboxEvent)
        assert event.aggregate_type == "document"
        assert event.aggregate_id == result.document_id
        assert event.event_type == OutboxEventType.DOCUMENT_PARSED
        assert event.payload["document_id"] == str(result.document_id)
        assert event.payload["version_id"] == str(result.version_id)
        assert event.payload["source_id"] == str(result.source_id)
        assert event.payload["tenant_id"] == "tenant-a"

    async def test_chunks_carry_document_and_version_ids(self, tmp_path: Any) -> None:
        workflow, _store, docs, _outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"word " * 500)

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )

        assert result.chunks_created > 0
        for chunk in docs.chunks:
            assert chunk.document_id == result.document_id
            assert chunk.version_id == result.version_id
            assert chunk.allowed_principals == ["engineering"]


class TestAcquireNode:
    async def test_missing_file_path_sets_error(self) -> None:
        workflow, store, docs, outbox = _build_workflow()

        result = await workflow.run(IngestionState(source_id=uuid4(), file_path=None))

        assert result.error == "file_path is required"
        assert result.raw_bytes is None
        assert result.document_id is None
        assert len(store.put_raw_calls) == 0
        assert len(docs.documents) == 0
        assert len(outbox.events) == 0

    async def test_nonexistent_file_sets_error(self) -> None:
        workflow, store, docs, _outbox = _build_workflow()

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path="/nonexistent/path/that/does/not/exist.txt",
                media_type="text/plain",
            )
        )

        assert result.error is not None
        assert "file not found" in result.error
        assert result.raw_bytes is None
        assert result.document_id is None
        assert len(store.put_raw_calls) == 0
        assert len(docs.documents) == 0


class TestParseNode:
    async def test_unsupported_media_type_propagates_error(self, tmp_path: Any) -> None:
        workflow, store, docs, outbox = _build_workflow()
        file_path = tmp_path / "doc.bin"
        file_path.write_bytes(b"binary blob")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="application/x-unknown-format",
            )
        )

        assert result.error is not None
        assert "unsupported media type" in result.error
        assert result.parsed is None
        assert result.document_id is None
        assert len(store.put_raw_calls) == 0
        assert len(docs.documents) == 0
        assert len(outbox.events) == 0


class TestChunkAndStoreNode:
    async def test_skips_when_no_source_id(self, tmp_path: Any) -> None:
        workflow, store, docs, outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Title\nBody")

        result = await workflow.run(
            IngestionState(
                source_id=None,
                file_path=str(file_path),
                media_type="text/plain",
            )
        )

        assert result.error is None
        assert result.document_id is None
        assert result.chunks_created == 0
        assert len(store.put_raw_calls) == 0
        assert len(docs.documents) == 0
        assert len(outbox.events) == 0

    async def test_skips_when_prior_error(self, tmp_path: Any) -> None:
        workflow, store, docs, outbox = _build_workflow()

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path="/definitely/does/not/exist.txt",
                media_type="text/plain",
            )
        )

        assert result.error is not None
        assert result.document_id is None
        assert result.chunks_created == 0
        assert len(store.put_raw_calls) == 0
        assert len(docs.documents) == 0
        assert len(outbox.events) == 0

    async def test_empty_file_yields_zero_chunks(self, tmp_path: Any) -> None:
        workflow, store, docs, _outbox = _build_workflow()
        file_path = tmp_path / "empty.txt"
        file_path.write_bytes(b"")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
            )
        )

        assert result.error is None
        assert result.document_id is not None
        assert result.chunks_created == 0
        assert len(store.put_raw_calls) == 1
        assert len(docs.documents) == 1
        assert len(docs.chunks) == 0


class TestEmitOutboxNode:
    async def test_skips_when_prior_error(self, tmp_path: Any) -> None:
        workflow, _store, _docs, outbox = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"data")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="application/x-unknown-format",
            )
        )

        assert result.error is not None
        assert result.outbox_event_id is None
        assert len(outbox.events) == 0


class TestStatePropagation:
    def test_initial_state_required_fields(self) -> None:
        state = IngestionState()
        assert state.source_id is None
        assert state.file_path is None
        assert state.media_type == "text/plain"
        assert state.tenant_id == ""
        assert state.allowed_principals == []
        assert state.raw_bytes is None
        assert state.parsed is None
        assert state.document_id is None
        assert state.version_id is None
        assert state.chunks_created == 0
        assert state.error is None
        assert state.outbox_event_id is None
        assert state.telemetry == {}

    async def test_uses_custom_chunker(self, tmp_path: Any) -> None:
        """Verify the workflow honors a caller-provided Chunker instance."""

        class _CountingChunker:
            def __init__(self) -> None:
                self.calls: list[Any] = []

            def chunk(
                self,
                content: Any,
                document_id: Any,
                version_id: Any,
                allowed_principals: Any,
            ) -> list[Chunk]:
                self.calls.append((content, document_id, version_id, allowed_principals))
                return [
                    Chunk(
                        chunk_id=uuid4(),
                        document_id=document_id,
                        version_id=version_id,
                        ordinal=0,
                        heading_path=[],
                        content="single",
                        token_count=1,
                        checksum="x",
                        start_locator="[0]",
                        end_locator="[0]",
                        allowed_principals=allowed_principals,
                    )
                ]

        custom_chunker: Any = _CountingChunker()
        docs: Any = _FakeDocumentRepository()
        outbox: Any = _FakeOutboxRepository()
        store: Any = _FakeObjectStore()
        workflow = IngestionWorkflow(
            documents=docs,
            outbox_repo=outbox,
            object_store=store,
            chunker=custom_chunker,
        )

        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"x")

        result = await workflow.run(
            IngestionState(
                source_id=uuid4(),
                file_path=str(file_path),
                media_type="text/plain",
                allowed_principals=["eng"],
            )
        )

        assert len(custom_chunker.calls) == 1
        assert result.chunks_created == 1
        assert len(docs.chunks) == 1
        assert docs.chunks[0].content == "single"

    def test_default_chunker_is_constructed_when_omitted(self) -> None:
        workflow, _, _, _ = _build_workflow()
        assert isinstance(workflow._chunker, Chunker)


class TestSha256Helper:
    def test_sha256_returns_known_hash(self) -> None:
        data = b"abc"
        assert IngestionWorkflow._sha256(data) == hashlib.sha256(b"abc").hexdigest()

    def test_sha256_empty_string(self) -> None:
        assert IngestionWorkflow._sha256(b"") == hashlib.sha256(b"").hexdigest()

    def test_sha256_is_deterministic(self) -> None:
        data = b"the quick brown fox"
        assert IngestionWorkflow._sha256(data) == IngestionWorkflow._sha256(data)


class TestGraphStructure:
    def test_graph_is_compiled(self) -> None:
        workflow, _, _, _ = _build_workflow()
        assert workflow._graph is not None

    async def test_pipeline_skips_emit_when_no_document_id(self) -> None:
        """If acquire fails, no chunks, no document, no outbox event."""
        workflow, _store, _docs, outbox = _build_workflow()
        result = await workflow.run(IngestionState(source_id=uuid4(), file_path=None))
        assert result.outbox_event_id is None
        assert result.document_id is None
        assert len(outbox.events) == 0
