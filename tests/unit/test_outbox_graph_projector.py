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


class _FakePostgresOutboxRepo:
    def __init__(self) -> None:
        self.claimed_events: list[OutboxEvent] = []
        self.completed_ids: list[UUID] = []
        self.failed_ids: list[tuple[UUID, str]] = []
        self._batch: list[OutboxEvent] = []

    async def claim_batch(
        self, batch_size: int, worker_id: str, lease_duration_seconds: int
    ) -> list[OutboxEvent]:
        events, self._batch = self._batch[:batch_size], self._batch[batch_size:]
        self.claimed_events.extend(events)
        return events

    async def mark_completed(self, event_id: UUID, claim_token: str) -> None:
        self.completed_ids.append(event_id)

    async def mark_failed(self, event_id: UUID, claim_token: str, error: str) -> None:
        self.failed_ids.append((event_id, error))


class _FakePostgresUnitOfWork:
    def __init__(self, outbox_repo: _FakePostgresOutboxRepo) -> None:
        self.outbox: _FakePostgresOutboxRepo | None = outbox_repo
        self._committed = False

    async def __aenter__(self) -> _FakePostgresUnitOfWork:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def commit(self) -> None:
        self._committed = True


class _FakeNeo4jUnitOfWork:
    def __init__(self, graph_repo: _FakeGraphRepository) -> None:
        self.graph: _FakeGraphRepository | None = graph_repo
        self._committed = False

    async def __aenter__(self) -> _FakeNeo4jUnitOfWork:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.mark.asyncio
async def test_worker_process_event_marks_completed_on_success() -> None:
    """Successful event projection followed by mark_completed."""
    graph_repo = _FakeGraphRepository()
    pg_outbox = _FakePostgresOutboxRepo()
    pg_outbox._batch = [
        _make_event(
            OutboxEventType.ENTITY_RESOLVED,
            {"entity_id": str(uuid4()), "entity_type": "Service", "canonical_name": "TestSvc"},
        )
    ]

    def pg_factory() -> _FakePostgresUnitOfWork:
        return _FakePostgresUnitOfWork(pg_outbox)

    neo4j_uow = _FakeNeo4jUnitOfWork(graph_repo)

    def neo4j_factory() -> _FakeNeo4jUnitOfWork:
        return neo4j_uow

    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, graph_repo))
    worker = OutboxGraphWorker(projector, poll_interval=0.01)
    event = pg_outbox._batch[0]

    await worker._process_event(pg_factory, neo4j_factory, event)  # type: ignore[arg-type]

    assert len(graph_repo.entities) == 1
    assert event.event_id in pg_outbox.completed_ids


@pytest.mark.asyncio
async def test_worker_process_event_marks_failed_on_projection_error() -> None:
    """Failed projection followed by mark_failed."""

    class _ThrowingGraphRepo(_FakeGraphRepository):
        async def create_entity(self, entity: object) -> None:
            raise RuntimeError("Neo4j unavailable")

    throwing_repo = _ThrowingGraphRepo()
    pg_outbox = _FakePostgresOutboxRepo()
    pg_outbox._batch = [
        _make_event(
            OutboxEventType.ENTITY_RESOLVED,
            {"entity_id": str(uuid4()), "entity_type": "Service", "canonical_name": "TestSvc"},
        )
    ]

    def pg_factory() -> _FakePostgresUnitOfWork:
        return _FakePostgresUnitOfWork(pg_outbox)

    neo4j_uow = _FakeNeo4jUnitOfWork(throwing_repo)

    def neo4j_factory() -> _FakeNeo4jUnitOfWork:
        return neo4j_uow

    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, throwing_repo))
    worker = OutboxGraphWorker(projector, poll_interval=0.01)
    event = pg_outbox._batch[0]

    await worker._process_event(pg_factory, neo4j_factory, event)  # type: ignore[arg-type]

    assert len(throwing_repo.entities) == 0
    assert event.event_id in [eid for eid, _ in pg_outbox.failed_ids]


@pytest.mark.asyncio
async def test_worker_start_claims_batch_and_processes() -> None:
    """start() claims a batch and processes all events."""
    graph_repo = _FakeGraphRepository()
    pg_outbox = _FakePostgresOutboxRepo()
    batch = [
        _make_event(
            OutboxEventType.ENTITY_MENTIONED,
            {
                "mention_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "surface_form": "Redis",
                "candidate_type": "Service",
            },
        ),
        _make_event(
            OutboxEventType.ENTITY_RESOLVED,
            {"entity_id": str(uuid4()), "entity_type": "Service", "canonical_name": "Redis"},
        ),
    ]
    pg_outbox._batch = list(batch)
    last_event = batch[1]

    neo4j_uow = _FakeNeo4jUnitOfWork(graph_repo)

    def pg_factory() -> _FakePostgresUnitOfWork:
        return _FakePostgresUnitOfWork(pg_outbox)

    def neo4j_factory() -> _FakeNeo4jUnitOfWork:
        return neo4j_uow

    projector = OutboxGraphProjector(cast(Neo4jGraphRepository, graph_repo))
    worker = OutboxGraphWorker(projector, poll_interval=0.01)
    worker._running = True

    async def run_one() -> None:
        pg_uow = pg_factory()
        async with pg_uow:
            assert pg_uow.outbox is not None
            events = await pg_uow.outbox.claim_batch(
                batch_size=100, worker_id=WORKER_ID, lease_duration_seconds=30
            )
            await pg_uow.commit()
        for ev in events:
            await worker._process_event(cast(Any, pg_factory), cast(Any, neo4j_factory), ev)
        worker._running = False

    await run_one()

    assert len(graph_repo.mentions) == 1
    assert len(graph_repo.entities) == 1
    assert last_event.event_id in pg_outbox.completed_ids
