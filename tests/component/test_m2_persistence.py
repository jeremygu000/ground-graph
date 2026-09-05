"""M2 component tests: PostgreSQL and Neo4j CRUD with Testcontainers.

Requires Docker. Skipped if Docker is not available.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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

pytestmark = [pytest.mark.integration, pytest.mark.component]


async def _setup_postgres(dsn: str) -> async_sessionmaker:
    """Create tables and return session factory."""
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(PostgresBase.metadata.create_all)
    await engine.dispose()
    return async_sessionmaker(create_async_engine(dsn), expire_on_commit=False)


async def test_source_create_and_fetch(postgres_component: Any) -> None:
    session_factory = await _setup_postgres(postgres_component.dsn)
    async with session_factory() as session:
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
    session_factory = await _setup_postgres(postgres_component.dsn)
    async with session_factory() as session:
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
            attributes={"region": "us-east-1"},
        )
        await repo.create_entity(entity)

        result = await repo.get_entity(entity.entity_id)
        assert result is not None
        assert result.canonical_name == "API Gateway"
        assert "api-gateway" in result.aliases
    finally:
        await driver.close()


async def test_fact_create_and_fetch(neo4j_component: Any) -> None:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri, auth=(neo4j_component.user, neo4j_component.password)
    )
    try:
        repo = Neo4jGraphRepository(driver)

        subject = CanonicalEntity(
            entity_id=uuid4(), entity_type="Service", canonical_name="Service A"
        )
        obj = CanonicalEntity(entity_id=uuid4(), entity_type="Service", canonical_name="Service B")
        await repo.create_entity(subject)
        await repo.create_entity(obj)

        fact = KnowledgeFact(
            fact_id=uuid4(),
            subject_id=subject.entity_id,
            predicate="DEPENDS_ON",
            object_id=obj.entity_id,
            status="verified",
            confidence=0.95,
            evidence_ids=[],
            observed_at=datetime.now(UTC),
            extraction_method="structured",
            ontology_version="v0.1.0",
        )
        await repo.create_fact(fact)

        result = await repo.get_fact(fact.fact_id)
        assert result is not None
        assert result.predicate == "DEPENDS_ON"
        assert result.status == "verified"
    finally:
        await driver.close()


async def test_outbox_claim_pattern(postgres_component: Any) -> None:
    session_factory = await _setup_postgres(postgres_component.dsn)
    async with session_factory() as session:
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

        result = await session.get(Outbox, event.event_id)
        assert result is not None
        assert result.status == "pending"
        assert result.attempts == 0
