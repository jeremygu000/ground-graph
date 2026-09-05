"""Unit tests for the LangGraph ingestion workflow.

Coverage target: src/groundgraph/workflows/ingestion_graph.py
The workflow now composes: ingest → emit-outbox.
ingest node delegates to IngestionService; emit-outbox writes an outbox event.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
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
        self.sources: dict[str, SourceDescriptor] = {}
        self._find_result: tuple[Any, Any] | None = None

    def set_source(self, source: SourceDescriptor) -> None:
        self.sources[str(source.source_id)] = source

    def set_find_result(self, result: tuple[Any, Any] | None) -> None:
        self._find_result = result

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
                return doc
        return None

    async def create_document(self, doc: ParsedDocument) -> None:
        self.documents.append(doc)

    async def create_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)


class _FakeOutboxRepository:
    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    async def add(self, event: OutboxEvent) -> OutboxEvent:
        self.events.append(event)
        return event


def _build_workflow() -> IngestionWorkflow:
    docs: Any = _FakeDocumentRepository()
    outbox: Any = _FakeOutboxRepository()
    store: Any = _FakeObjectStore()
    service = IngestionService(documents=docs, object_store=store, chunker=Chunker())
    workflow = IngestionWorkflow(ingestion_service=service, outbox_repo=outbox)
    workflow._docs = docs
    workflow._outbox = outbox
    workflow._store = store
    return workflow


class TestIngestionWorkflowHappyPath:
    async def test_runs_full_pipeline_end_to_end(self, tmp_path: Any) -> None:
        source_id = uuid4()
        workflow = _build_workflow()
        workflow._docs.set_source(
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

        result = await workflow.run(
            IngestionState(
                source_id=source_id,
                file_path=str(file_path),
                media_type="text/plain",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )

        assert result.document_id is not None
        assert result.version_id is not None
        assert len(workflow._docs.documents) == 1
        assert len(workflow._docs.chunks) > 0
        assert len(workflow._store.put_raw_calls) == 1
        assert len(workflow._outbox.events) == 1
        assert result.error is None


class TestEmitOutboxNode:
    async def test_emits_outbox_event(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        source_id = uuid4()
        workflow._docs.set_source(
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

        result = await workflow.run(
            IngestionState(
                source_id=source_id,
                file_path=str(file_path),
                media_type="text/plain",
            )
        )

        assert len(workflow._outbox.events) == 1
        event = workflow._outbox.events[0]
        assert event.aggregate_type == "document"
        assert event.event_type == OutboxEventType.DOCUMENT_PARSED
        assert result.outbox_event_id is not None

    async def test_outbox_event_contains_document_metadata(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        source_id = uuid4()
        workflow._docs.set_source(
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

        await workflow.run(
            IngestionState(source_id=source_id, file_path=str(file_path), media_type="text/plain")
        )

        event = workflow._outbox.events[0]
        payload = dict(event.payload)
        assert "document_id" in payload
        assert "version_id" in payload
        assert "source_id" in payload


class TestStatePropagation:
    async def test_telemetry_accumulates(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        source_id = uuid4()
        workflow._docs.set_source(
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

        result = await workflow.run(
            IngestionState(
                source_id=source_id,
                file_path=str(file_path),
                media_type="text/plain",
            )
        )

        assert "outbox_emitted" in result.telemetry


class TestSha256Helper:
    def test_sha256_returns_hex_string(self) -> None:
        result = IngestionWorkflow._sha256(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestGraphStructure:
    def test_workflow_has_two_nodes(self) -> None:
        workflow = _build_workflow()
        assert {"ingest", "emit_outbox"}.issubset(set(workflow._graph.nodes))


class TestErrorPaths:
    async def test_missing_source_id_returns_error(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello")

        result = await workflow.run(
            IngestionState(source_id=None, file_path=str(file_path), media_type="text/plain")
        )
        assert result.error is not None

    async def test_unknown_source_returns_error(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello")

        result = await workflow.run(
            IngestionState(source_id=uuid4(), file_path=str(file_path), media_type="text/plain")
        )
        assert result.error is not None
        assert "source not found" in result.error

    async def test_outbox_not_called_when_ingest_fails(self, tmp_path: Any) -> None:
        workflow = _build_workflow()
        file_path = tmp_path / "doc.txt"
        file_path.write_bytes(b"Hello")

        result = await workflow.run(
            IngestionState(source_id=uuid4(), file_path=str(file_path), media_type="text/plain")
        )
        assert result.error is not None
        assert len(workflow._outbox.events) == 0
