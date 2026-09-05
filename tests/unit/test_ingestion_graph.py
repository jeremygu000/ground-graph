"""Unit tests for the LangGraph ingestion workflow.

Coverage target: src/groundgraph/workflows/ingestion_graph.py
Single-node workflow: ingest — delegates to IngestionService.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.services import IngestionService
from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.domain.evidence import OutboxEvent
from groundgraph.workflows.ingestion_graph import IngestionState, IngestionWorkflow


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


def _build_workflow() -> tuple[IngestionWorkflow, _FakeDocumentRepository, _FakeOutboxRepository]:
    docs: Any = _FakeDocumentRepository()
    outbox: Any = _FakeOutboxRepository()
    store: Any = _FakeObjectStore()
    service = IngestionService(
        documents=docs, object_store=store, outbox_repo=outbox, chunker=Chunker()
    )
    workflow = IngestionWorkflow(ingestion_service=service)
    return workflow, docs, outbox


class TestWorkflowHappyPath:
    async def test_runs_full_pipeline(self, tmp_path: Any) -> None:
        workflow, docs, outbox = _build_workflow()
        source_id = uuid4()
        docs.set_source(
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
                media_type="text/markdown",
            )
        )

        assert result.document_id is not None
        assert result.version_id is not None
        assert result.created_new_version is True
        assert result.error is None
        assert len(docs.documents) == 1
        assert len(docs.chunks) > 0
        assert len(outbox.events) == 1

    async def test_idempotent_no_new_version(self, tmp_path: Any) -> None:
        workflow, docs, _ = _build_workflow()
        source_id = uuid4()
        docs.set_source(
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

        result1 = await workflow.run(
            IngestionState(
                source_id=source_id,
                file_path=str(file_path),
                media_type="text/plain",
            )
        )
        docs.set_find_result((result1.document_id, result1.version_id))

        result2 = await workflow.run(
            IngestionState(
                source_id=source_id,
                file_path=str(file_path),
                media_type="text/plain",
            )
        )

        assert result2.document_id == result1.document_id
        assert result2.version_id == result1.version_id
        assert result2.created_new_version is False


class TestWorkflowErrorPaths:
    async def test_missing_source_id_returns_error(self, tmp_path: Any) -> None:
        workflow, _, _ = _build_workflow()
        result = await workflow.run(
            IngestionState(
                source_id=None, file_path=str(tmp_path / "x.txt"), media_type="text/plain"
            )
        )
        assert result.error is not None

    async def test_unknown_source_returns_error(self, tmp_path: Any) -> None:
        workflow, _, _ = _build_workflow()
        result = await workflow.run(
            IngestionState(
                source_id=uuid4(), file_path=str(tmp_path / "x.txt"), media_type="text/plain"
            )
        )
        assert result.error is not None
        assert "source not found" in result.error

    async def test_outbox_not_emitted_on_failure(self, tmp_path: Any) -> None:
        workflow, _, outbox = _build_workflow()
        result = await workflow.run(
            IngestionState(
                source_id=uuid4(), file_path=str(tmp_path / "x.txt"), media_type="text/plain"
            )
        )
        assert result.error is not None
        assert len(outbox.events) == 0


class TestGraphStructure:
    def test_workflow_has_single_node(self) -> None:
        workflow, _, _ = _build_workflow()
        assert set(workflow._graph.nodes).issubset({"ingest", "__start__"})
