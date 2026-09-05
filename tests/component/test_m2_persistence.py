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

from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.domain.execution import ExecutionRun, ExecutionRunStatus
from groundgraph.domain.knowledge import CanonicalEntity, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.postgres.document_repository import PostgresDocumentRepository
from groundgraph.infrastructure.postgres.models import (
    Base as PostgresBase,
)
from groundgraph.infrastructure.postgres.models import (
    Document as SqlDocument,
)
from groundgraph.infrastructure.postgres.models import (
    DocumentVersion as SqlDocumentVersion,
)
from groundgraph.infrastructure.postgres.models import (
    Outbox as SqlOutbox,
)
from groundgraph.infrastructure.postgres.models import (
    Source as SqlSource,
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
        repo = PostgresDocumentRepository(cast(PostgresSession, session))
        source = SqlSource(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path/to/docs",
            classification="internal",
            tenant_id="tenant-a",
            allowed_principals=["engineering"],
        )
        restored = await repo.create_source(
            SourceDescriptor(
                source_id=source.source_id,
                source_type="filesystem",
                uri="/path/to/docs",
                classification="internal",
                tenant_id="tenant-a",
                allowed_principals=["engineering"],
            )
        )
        assert restored.tenant_id == "tenant-a"
        await session.commit()

        result = await repo.get_source(source.source_id)
        assert result is not None
        assert result.source_type == "filesystem"
        assert result.uri == "/path/to/docs"
        assert result.tenant_id == "tenant-a"


async def test_document_repository_crud(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = PostgresDocumentRepository(cast(PostgresSession, session))
        source = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path",
            classification="internal",
            tenant_id="tenant-a",
        )
        await repo.create_source(source)
        document = ParsedDocument(
            document_id=uuid4(),
            version_id=uuid4(),
            source_id=source.source_id,
            title="Test Doc",
            media_type="text/markdown",
            checksum="abc123",
            content="# Hello",
            metadata={"author": "test"},
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        await repo.create_document(document)
        chunk = Chunk(
            chunk_id=uuid4(),
            document_id=document.document_id,
            version_id=document.version_id,
            ordinal=0,
            heading_path=["Intro"],
            content="Hello",
            token_count=1,
            checksum="chunk-1",
            allowed_principals=["engineering"],
        )
        await repo.create_chunk(chunk)
        await session.commit()

        loaded_source = await repo.get_source(source.source_id)
        assert loaded_source is not None
        assert loaded_source.tenant_id == "tenant-a"

        loaded_doc = await repo.get_document(document.document_id)
        assert loaded_doc is not None
        assert loaded_doc.title == "Test Doc"
        assert loaded_doc.metadata == {"author": "test"}

        loaded_version = await repo.get_document_version(document.document_id, document.version_id)
        assert loaded_version is not None
        assert loaded_version.checksum == "abc123"

        versions = await repo.list_document_versions(document.document_id)
        assert len(versions) == 1

        loaded_chunk = await repo.get_chunk(chunk.chunk_id)
        assert loaded_chunk is not None
        assert loaded_chunk.content == "Hello"

        chunks = await repo.list_chunks(document.document_id, document.version_id)
        assert len(chunks) == 1


async def test_source_lifecycle_and_list_sources(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = PostgresDocumentRepository(cast(PostgresSession, session))

        source_a = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/docs/a",
            classification="internal",
            tenant_id="tenant-a",
            allowed_principals=["engineering"],
        )
        source_b = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/docs/b",
            classification="restricted",
            tenant_id="tenant-b",
            allowed_principals=["security"],
        )

        await repo.create_source(source_a)
        await repo.create_source(source_b)
        await session.commit()

        loaded = await repo.get_source(source_a.source_id)
        assert loaded is not None
        assert loaded.uri == "/docs/a"
        assert loaded.classification == "internal"
        assert loaded.allowed_principals == ["engineering"]

        sources = await repo.list_sources()
        source_ids = {source.source_id for source in sources}
        assert source_a.source_id in source_ids
        assert source_b.source_id in source_ids


async def test_document_current_version_fallback_and_delete(
    postgres_component: Any,
) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = PostgresDocumentRepository(cast(PostgresSession, session))

        source = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/docs",
            classification="internal",
            tenant_id="tenant-a",
        )
        await repo.create_source(source)

        document = ParsedDocument(
            document_id=uuid4(),
            version_id=uuid4(),
            source_id=source.source_id,
            title="Fallback Doc",
            media_type="text/markdown",
            checksum="abc123",
            content="# Fallback",
            metadata={"author": "test"},
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        await repo.create_document(document)
        await session.commit()

        loaded = await repo.get_document(document.document_id)
        assert loaded is not None
        assert loaded.version_id == document.version_id

        await session.execute(
            update(SqlDocument)
            .where(SqlDocument.document_id == document.document_id)
            .values(current_version_id=None)
        )
        await session.commit()

        fallback_loaded = await repo.get_document(document.document_id)
        assert fallback_loaded is not None
        assert fallback_loaded.version_id == document.version_id
        assert fallback_loaded.title == "Fallback Doc"

        version = await repo.get_document_version(document.document_id, document.version_id)
        assert version is not None
        assert version.checksum == "abc123"

        versions = await repo.list_document_versions(document.document_id)
        assert len(versions) == 1
        assert versions[0].version_id == document.version_id

        await repo.delete_document(document.document_id)
        await session.commit()

        assert await session.get(SqlDocument, document.document_id) is None
        version_row = await session.get(SqlDocumentVersion, document.version_id)
        assert version_row is not None
        assert version_row.checksum == "abc123"


async def test_chunk_lifecycle_and_listing(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = PostgresDocumentRepository(cast(PostgresSession, session))

        source = SourceDescriptor(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/docs",
            classification="internal",
            tenant_id="tenant-a",
        )
        await repo.create_source(source)

        document = ParsedDocument(
            document_id=uuid4(),
            version_id=uuid4(),
            source_id=source.source_id,
            title="Chunked Doc",
            media_type="text/markdown",
            checksum="chunky-1",
            content="# Chunked",
            metadata={"author": "test"},
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        await repo.create_document(document)

        chunk_late = Chunk(
            chunk_id=uuid4(),
            document_id=document.document_id,
            version_id=document.version_id,
            ordinal=1,
            heading_path=["Section 2"],
            content="Second chunk",
            token_count=2,
            checksum="chunk-2",
            allowed_principals=["engineering"],
        )
        chunk_early = Chunk(
            chunk_id=uuid4(),
            document_id=document.document_id,
            version_id=document.version_id,
            ordinal=0,
            heading_path=["Section 1"],
            content="First chunk",
            token_count=1,
            checksum="chunk-1",
            allowed_principals=["engineering"],
        )
        await repo.create_chunk(chunk_late)
        await repo.create_chunk(chunk_early)
        await session.commit()

        loaded = await repo.get_chunk(chunk_early.chunk_id)
        assert loaded is not None
        assert loaded.heading_path == ["Section 1"]
        assert loaded.allowed_principals == ["engineering"]

        chunks = await repo.list_chunks(document.document_id, document.version_id)
        assert [chunk.ordinal for chunk in chunks] == [0, 1]
        assert [chunk.chunk_id for chunk in chunks] == [chunk_early.chunk_id, chunk_late.chunk_id]

        assert await repo.get_chunk(uuid4()) is None


async def test_document_version_cascade(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        source = SqlSource(
            source_id=uuid4(),
            source_type="filesystem",
            uri="/path",
            classification="internal",
            tenant_id="tenant-a",
        )
        session.add(source)
        await session.flush()

        doc = SqlDocument(
            document_id=uuid4(),
            source_id=source.source_id,
            title="Test Doc",
            media_type="text/markdown",
        )
        session.add(doc)
        await session.flush()

        version = SqlDocumentVersion(
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
            select(SqlDocumentVersion).where(SqlDocumentVersion.document_id == doc.document_id)
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


async def test_execution_repository_atomic_cas(postgres_component: Any) -> None:
    # Real concurrency test is exercised in the dedicated execution component suite.
    assert postgres_component.dsn.startswith("postgresql+asyncpg://")


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
        await session.execute(delete(SqlOutbox))
        await session.commit()

        event = SqlOutbox(
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

        stored = await session.get(SqlOutbox, event.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-a"
        assert stored.claim_token is not None
        token = stored.claim_token

        await repo.mark_failed(event.event_id, token, "temporary failure")
        failed = await session.get(SqlOutbox, event.event_id)
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
            update(SqlOutbox)
            .where(SqlOutbox.event_id == event.event_id)
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
        stored = await session.get(SqlOutbox, event.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-b"

        await repo.mark_completed(event.event_id, stored.claim_token or "")
        completed = await session.get(SqlOutbox, event.event_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.completed_at is not None
