"""Unit tests for outbox graph projector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.postgres.outbox_graph_projector import (
    WORKER_ID,
    OutboxGraphProjector,
    OutboxGraphWorker,
)


class _FakeGraphRepository:
    def __init__(self) -> None:
        self.mentions: list[object] = []
        self.entities: list[object] = []
        self.facts: list[object] = []

    async def create_mention(self, mention: object) -> None:
        self.mentions.append(mention)

    async def create_entity(self, entity: object) -> None:
        self.entities.append(entity)

    async def create_fact(self, fact: object) -> None:
        self.facts.append(fact)


def _make_event(
    event_type: OutboxEventType,
    payload: dict[str, Any],
    event_id: UUID | None = None,
    claim_token: str | None = None,
) -> OutboxEvent:
    return OutboxEvent(
        event_id=event_id or uuid4(),
        aggregate_type="test",
        aggregate_id=uuid4(),
        event_type=event_type,
        payload=payload,
        status=OutboxEventStatus.CLAIMED,
        attempts=0,
        created_at=datetime.now(UTC),
        claim_token=claim_token or uuid4().hex,
    )


@pytest.mark.asyncio
async def test_project_mention_event() -> None:
    """ENTITY_MENTIONED events are projected to Neo4j."""
    repo = _FakeGraphRepository()
    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, repo))

    mention_id = uuid4()
    chunk_id = uuid4()
    event = _make_event(
        OutboxEventType.ENTITY_MENTIONED,
        {
            "mention_id": str(mention_id),
            "chunk_id": str(chunk_id),
            "surface_form": "PostgreSQL",
            "candidate_type": "Database",
            "locator": None,
            "extraction_confidence": 0.9,
        },
    )

    await projector.project_event(event)

    assert len(repo.mentions) == 1


@pytest.mark.asyncio
async def test_project_entity_event() -> None:
    """ENTITY_RESOLVED events are projected to Neo4j."""
    repo = _FakeGraphRepository()
    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, repo))

    entity_id = uuid4()
    event = _make_event(
        OutboxEventType.ENTITY_RESOLVED,
        {
            "entity_id": str(entity_id),
            "entity_type": "Database",
            "canonical_name": "PostgreSQL",
            "aliases": ["PostgreSQL"],
            "attributes": {},
        },
    )

    await projector.project_event(event)

    assert len(repo.entities) == 1
    assert repo.entities[0].canonical_name == "PostgreSQL"


@pytest.mark.asyncio
async def test_project_fact_event() -> None:
    """FACT_CANDIDATE events are projected to Neo4j."""
    repo = _FakeGraphRepository()
    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, repo))

    fact_id = uuid4()
    subject_id = uuid4()
    object_id = uuid4()
    now = datetime.now(UTC)
    event = _make_event(
        OutboxEventType.FACT_CANDIDATE,
        {
            "fact_id": str(fact_id),
            "subject_id": str(subject_id),
            "predicate": "depends_on",
            "object_id": str(object_id),
            "status": "candidate",
            "confidence": 0.8,
            "evidence_ids": [],
            "valid_from": None,
            "valid_to": None,
            "observed_at": now.isoformat(),
            "extraction_method": "llm",
            "ontology_version": "v0.1.0",
            "tenant_id": "test-tenant",
            "allowed_principals": ["eng"],
        },
    )

    await projector.project_event(event)

    assert len(repo.facts) == 1
    assert repo.facts[0].predicate == "depends_on"


@pytest.mark.asyncio
async def test_project_unknown_event_type_logs_warning() -> None:
    """Unknown event types are logged but do not raise."""
    repo = _FakeGraphRepository()
    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, repo))

    event = _make_event(OutboxEventType.DOCUMENT_PARSED, {})

    await projector.project_event(event)

    assert len(repo.mentions) == 0
    assert len(repo.entities) == 0
    assert len(repo.facts) == 0


@pytest.mark.asyncio
async def test_project_event_handles_exception() -> None:
    """Exception in _project_mention is logged and re-raised."""

    class _ThrowingRepo(_FakeGraphRepository):
        async def create_mention(self, mention: object) -> None:
            raise RuntimeError("DB error")

    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, _ThrowingRepo()))

    event = _make_event(
        OutboxEventType.ENTITY_MENTIONED,
        {
            "mention_id": str(uuid4()),
            "chunk_id": str(uuid4()),
            "surface_form": "PostgreSQL",
            "candidate_type": "Database",
            "locator": None,
            "extraction_confidence": 0.9,
        },
    )

    with pytest.raises(RuntimeError, match="DB error"):
        await projector.project_event(event)


def test_worker_stop() -> None:
    """Worker can be stopped."""
    repo = _FakeGraphRepository()
    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, repo))

    worker = OutboxGraphWorker(projector, poll_interval=0.01)
    assert worker._running is False
    worker.stop()
    assert worker._running is False


def test_worker_id_constant() -> None:
    """Worker uses a stable worker_id constant for lease tracking."""
    assert WORKER_ID == "graph-projector"
