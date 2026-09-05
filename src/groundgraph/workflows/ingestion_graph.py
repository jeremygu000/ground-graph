"""LangGraph ingestion workflow (plan.md §6 + M3 tasks).

Composes: acquire → parse → chunk → store → emit-outbox.
Each node calls application services; no infrastructure imports.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from groundgraph.application.ingestion.services import IngestionService
from groundgraph.application.ports import OutboxRepository
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType


@dataclass
class IngestionState:
    source_id: UUID | None = None
    file_path: str | None = None
    media_type: str = "text/plain"
    tenant_id: str = ""
    allowed_principals: list[str] = field(default_factory=list[str])

    raw_bytes: bytes | None = None
    document_id: UUID | None = None
    version_id: UUID | None = None
    chunks_created: int = 0

    error: str | None = None
    outbox_event_id: UUID | None = None

    telemetry: dict[str, Any] = field(default_factory=dict)


class IngestionWorkflow:
    def __init__(
        self,
        ingestion_service: IngestionService,
        outbox_repo: OutboxRepository,
    ) -> None:
        self._ingestion = ingestion_service
        self._outbox = outbox_repo
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        g = StateGraph(IngestionState)

        g.add_node("ingest", self._ingest)
        g.add_node("emit_outbox", self._emit_outbox)

        g.add_edge(START, "ingest")
        g.add_edge("ingest", "emit_outbox")
        g.add_edge("emit_outbox", END)

        return g.compile()

    async def _ingest(self, state: IngestionState) -> dict[str, Any]:
        if state.source_id is None:
            return {"error": "source_id is required"}
        if not state.file_path:
            return {"error": "file_path is required"}

        try:
            doc_id, ver_id = await self._ingestion.ingest_file(
                source_id=state.source_id,
                file_path=state.file_path,
                media_type=state.media_type,
            )
        except Exception as exc:
            return {"error": str(exc)}
        else:
            return {
                "document_id": doc_id,
                "version_id": ver_id,
                "telemetry": {**state.telemetry, "outbox_emitted": False},
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
