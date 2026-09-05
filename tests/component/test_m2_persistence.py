"""M2 component tests: PostgreSQL and Neo4j CRUD with Testcontainers.

Requires Docker. Skipped if Docker is not available.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase
from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from groundgraph.domain.execution import ExecutionRun, ExecutionRunStatus
from groundgraph.domain.knowledge import CanonicalEntity, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.postgres.models import (
    Base as PostgresBase,
)
from groundgraph.infrastructure.postgres.models import (
    Document,
    DocumentVersion,
    Outbox,
    Source,
)
from groundgraph.infrastructure.postgres.outbox_repository import (
    PostgresOutboxRepository,
)
from groundgraph.infrastructure.postgres.session import PostgresSession

pytestmark = [pytest.mark.integration, pytest.mark.component]


@asynccontextmanager
async def _setup_postgres(
    dsn: str,
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create tables and yield a session factory."""
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(PostgresBase.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_source_create_and_fetch(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        source = Source(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path/to/docs",
            classification="internal",
            allowed_principals=["engineering"],
        )
        session.add(source)
        await session.commit()

        result = await session.get(Source, source.source_id)
        assert result is not None
        assert result.source_type == "filesystem"
        assert result.uri == "/path/to/docs"


async def test_document_version_cascade(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        source = Source(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path",
            classification="internal",
        )
        session.add(source)
        await session.flush()

        doc = Document(
            document_id=uuid4(),
            source_id=source.source_id,
            title="Test Doc",
            media_type="text/markdown",
        )
        session.add(doc)
        await session.flush()

        version = DocumentVersion(
            version_id=uuid4(),
            document_id=doc.document_id,
            checksum="abc123",
            content="# Hello",
            doc_metadata={"author": "test"},
            is_current=True,
        )
        session.add(version)
        await session.commit()

        count_result = await session.execute(
            select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
        )
        versions = count_result.scalars().all()
        assert len(versions) == 1
        assert versions[0].checksum == "abc123"


async def test_execution_run_state_machine(postgres_component: Any) -> None:
    run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal="engineering",
        tenant_id="default",
    )
    assert run.status == ExecutionRunStatus.PENDING


async def test_entity_create_and_fetch(neo4j_component: Any) -> None:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        repo = Neo4jGraphRepository(driver)

        entity = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="API Gateway",
            aliases=["api-gateway", "gateway"],
            attributes={
                "region": "us-east-1",
                "owner": {"team": "platform", "contacts": ["alice", "bob"]},
            },
        )
        await repo.create_entity(entity)

        result = await repo.get_entity(entity.entity_id)
        assert result is not None
        assert result.canonical_name == "API Gateway"
        assert "api-gateway" in result.aliases
        assert result.attributes == {
            "region": "us-east-1",
            "owner": {"team": "platform", "contacts": ["alice", "bob"]},
        }
    finally:
        await driver.close()


async def test_fact_create_and_fetch(neo4j_component: Any) -> None:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        repo = Neo4jGraphRepository(driver)

        subject = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="Service A",
            aliases=[],
            attributes={},
        )
        obj = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="Service B",
            aliases=[],
            attributes={},
        )
        await repo.create_entity(subject)
        await repo.create_entity(obj)

        valid_from = datetime(2024, 1, 1, tzinfo=UTC)
        valid_to = datetime(2024, 12, 31, tzinfo=UTC)
        fact = KnowledgeFact(
            fact_id=uuid4(),
            subject_id=subject.entity_id,
            predicate="DEPENDS_ON",
            object_id=obj.entity_id,
            status="verified",
            confidence=0.95,
            evidence_ids=[uuid4()],
            valid_from=valid_from,
            valid_to=valid_to,
            observed_at=datetime.now(UTC),
            extraction_method="structured",
            ontology_version="v0.1.0",
        )
        await repo.create_fact(fact)

        result = await repo.get_fact(fact.fact_id)
        assert result is not None
        assert result.predicate == "DEPENDS_ON"
        assert result.status == "verified"
        assert result.valid_from == valid_from
        assert result.valid_to == valid_to
    finally:
        await driver.close()


async def test_find_entities_returns_entity(neo4j_component: Any) -> None:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        repo = Neo4jGraphRepository(driver)

        entity = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="FindMe Service",
            aliases=["findme"],
            attributes={"env": "prod"},
        )
        await repo.create_entity(entity)

        results = await repo.find_entities(canonical_name="FindMe Service")
        assert len(results) == 1
        assert results[0].canonical_name == "FindMe Service"
        assert results[0].entity_type == "Service"
    finally:
        await driver.close()


async def test_find_facts_datetime_conversion(neo4j_component: Any) -> None:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        repo = Neo4jGraphRepository(driver)

        subject = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="Source Svc",
            aliases=[],
            attributes={},
        )
        obj = CanonicalEntity(
            entity_id=uuid4(),
            entity_type="Service",
            canonical_name="Target Svc",
            aliases=[],
            attributes={},
        )
        await repo.create_entity(subject)
        await repo.create_entity(obj)

        valid_from = datetime(2024, 1, 1, tzinfo=UTC)
        valid_to = datetime(2024, 12, 31, tzinfo=UTC)
        observed = datetime(2024, 6, 15, tzinfo=UTC)
        fact = KnowledgeFact(
            fact_id=uuid4(),
            subject_id=subject.entity_id,
            predicate="CALLS",
            object_id=obj.entity_id,
            status="verified",
            confidence=0.99,
            evidence_ids=[uuid4()],
            valid_from=valid_from,
            valid_to=valid_to,
            observed_at=observed,
            extraction_method="llm",
            ontology_version="v0.1.0",
        )
        await repo.create_fact(fact)

        results = await repo.find_facts(subject_id=subject.entity_id)
        assert len(results) == 1
        result = results[0]
        assert result.predicate == "CALLS"
        assert result.valid_from == valid_from
        assert result.valid_to == valid_to
        assert result.observed_at == observed
    finally:
        await driver.close()


async def test_outbox_claim_pattern(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        await session.execute(delete(Outbox))
        await session.commit()

        event = Outbox(
            event_id=uuid4(),
            aggregate_type="document",
            aggregate_id=uuid4(),
            event_type="document_parsed",
            payload={"title": "Test"},
            status="pending",
            attempts=0,
            created_at=datetime.now(UTC),
        )
        session.add(event)
        await session.commit()

        repo = PostgresOutboxRepository(cast(PostgresSession, session))
        claimed = await repo.claim_batch(
            batch_size=1,
            worker_id="worker-a",
            lease_duration_seconds=30,
        )
        assert len(claimed) == 1
        claimed_event = claimed[0]
        assert claimed_event.event_id == event.event_id
        assert claimed_event.status.value == "claimed"
        assert claimed_event.attempts == 1
        assert claimed_event.claimed_at is not None

        stored = await session.get(Outbox, event.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-a"
        assert stored.claim_token is not None
        token = stored.claim_token

        await repo.mark_failed(event.event_id, token, "temporary failure")
        failed = await session.get(Outbox, event.event_id)
        assert failed is not None
        assert failed.status == "pending"
        assert failed.attempts == 1
        assert failed.last_error == "temporary failure"
        assert failed.claim_token is None
        assert failed.claimed_by is None

        second_claim = await repo.claim_batch(
            batch_size=1,
            worker_id="worker-b",
            lease_duration_seconds=30,
        )
        assert second_claim == []

        await session.execute(
            update(Outbox)
            .where(Outbox.event_id == event.event_id)
            .values(available_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()

        third_claim = await repo.claim_batch(
            batch_size=1,
            worker_id="worker-b",
            lease_duration_seconds=30,
        )
        assert len(third_claim) == 1
        assert third_claim[0].event_id == event.event_id
        stored = await session.get(Outbox, event.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-b"

        await repo.mark_completed(event.event_id, stored.claim_token or "")
        completed = await session.get(Outbox, event.event_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.completed_at is not None
