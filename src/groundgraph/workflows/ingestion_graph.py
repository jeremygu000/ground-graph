"""LangGraph ingestion workflow (plan.md §6 + M3 tasks).

Composes: acquire → parse → chunk → store → emit-outbox.
Each node calls application services; no infrastructure imports.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.parsers import ParsedContent, ParserRegistry
from groundgraph.application.ports import DocumentRepository, ObjectStore, OutboxRepository
from groundgraph.domain.documents import ParsedDocument
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType


@dataclass
class IngestionState:
    source_id: UUID | None = None
    file_path: str | None = None
    media_type: str = "text/plain"
    tenant_id: str = ""
    allowed_principals: list[str] = field(default_factory=list[str])

    raw_bytes: bytes | None = None
    parsed: ParsedContent | None = None
    document_id: UUID | None = None
    version_id: UUID | None = None
    chunks_created: int = 0

    error: str | None = None
    outbox_event_id: UUID | None = None

    telemetry: dict[str, Any] = field(default_factory=dict)


class IngestionWorkflow:
    def __init__(
        self,
        documents: DocumentRepository,
        outbox_repo: OutboxRepository,
        object_store: ObjectStore,
        chunker: Chunker | None = None,
    ) -> None:
        self._documents = documents
        self._outbox = outbox_repo
        self._object_store = object_store
        self._chunker = chunker or Chunker()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        g = StateGraph(IngestionState)

        g.add_node("acquire", self._acquire)
        g.add_node("parse", self._parse)
        g.add_node("chunk_and_store", self._chunk_and_store)
        g.add_node("emit_outbox", self._emit_outbox)

        g.add_edge(START, "acquire")
        g.add_edge("acquire", "parse")
        g.add_edge("parse", "chunk_and_store")
        g.add_edge("chunk_and_store", "emit_outbox")
        g.add_edge("emit_outbox", END)

        return g.compile()

    async def _acquire(self, state: IngestionState) -> dict[str, Any]:
        if not state.file_path:
            return {"error": "file_path is required"}
        path = Path(state.file_path)

        def _exists() -> bool:
            return path.exists()

        if not await asyncio.to_thread(_exists):
            return {"error": f"file not found: {state.file_path}"}

        def _read() -> bytes:
            return path.read_bytes()

        raw_bytes = await asyncio.to_thread(_read)
        return {
            "raw_bytes": raw_bytes,
            "telemetry": {**state.telemetry, "acquire_bytes": len(raw_bytes)},
        }

    async def _parse(self, state: IngestionState) -> dict[str, Any]:
        if state.error or state.raw_bytes is None:
            return {}
        parser = ParserRegistry.get(state.media_type)
        if parser is None:
            return {"error": f"unsupported media type: {state.media_type}"}

        parsed = parser.parse(state.raw_bytes)
        return {
            "parsed": parsed,
            "telemetry": {**state.telemetry, "parse_tokens": int(len(parsed.body) * 0.25)},
        }

    async def _chunk_and_store(self, state: IngestionState) -> dict[str, Any]:
        if state.error or state.parsed is None or not state.source_id:
            return {}

        document_id = uuid4()
        version_id = uuid4()
        parsed = state.parsed

        raw_key = f"sources/{state.source_id}/{document_id}/v1/raw"
        await self._object_store.put_raw(raw_key, state.raw_bytes or b"", state.media_type)

        document = ParsedDocument(
            document_id=document_id,
            version_id=version_id,
            source_id=state.source_id,
            title=parsed.title,
            media_type=state.media_type,
            checksum=self._sha256(state.raw_bytes or b""),
            content=parsed.body,
            metadata={"file_path": state.file_path or "", **parsed.metadata},
            effective_at=datetime.now(UTC),
        )
        await self._documents.create_document(document)

        chunks = self._chunker.chunk(
            content=parsed,
            document_id=document_id,
            version_id=version_id,
            allowed_principals=state.allowed_principals,
        )
        for chunk in chunks:
            await self._documents.create_chunk(chunk)

        return {
            "document_id": document_id,
            "version_id": version_id,
            "chunks_created": len(chunks),
            "telemetry": {
                **state.telemetry,
                "chunks_created": len(chunks),
            },
        }

    async def _emit_outbox(self, state: IngestionState) -> dict[str, Any]:
        if state.error or not state.document_id or not state.version_id:
            return {}
        event_id = uuid4()
        event = OutboxEvent(
            event_id=event_id,
            aggregate_type="document",
            aggregate_id=state.document_id,
            event_type=OutboxEventType.DOCUMENT_PARSED,
            payload={
                "document_id": str(state.document_id),
                "version_id": str(state.version_id),
                "source_id": str(state.source_id),
                "tenant_id": state.tenant_id,
            },
            created_at=datetime.now(UTC),
        )
        await self._outbox.add(event)
        return {
            "outbox_event_id": event_id,
            "telemetry": {**state.telemetry, "outbox_emitted": True},
        }

    async def run(self, state: IngestionState) -> IngestionState:
        result = await self._graph.ainvoke(state)
        return IngestionState(**result) if isinstance(result, dict) else result

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
