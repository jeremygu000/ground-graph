"""Component tests for the PostgreSQL outbox repository."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.infrastructure.postgres.models import Base as PostgresBase
from groundgraph.infrastructure.postgres.models import Outbox
from groundgraph.infrastructure.postgres.outbox_repository import PostgresOutboxRepository
from groundgraph.infrastructure.postgres.session import PostgresSession

pytestmark = [pytest.mark.integration, pytest.mark.component]


@asynccontextmanager
async def _setup_postgres(dsn: str) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(PostgresBase.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _truncate_outbox(session: AsyncSession) -> None:
    await session.execute(text("TRUNCATE TABLE outbox RESTART IDENTITY CASCADE"))
    await session.commit()


async def _insert_event(
    session: AsyncSession,
    *,
    status: OutboxEventStatus = OutboxEventStatus.PENDING,
    available_at: datetime | None = None,
    created_at: datetime | None = None,
) -> Outbox:
    now = created_at or datetime.now(UTC)
    row = Outbox(
        event_id=uuid4(),
        aggregate_type="document",
        aggregate_id=uuid4(),
        event_type=OutboxEventType.DOCUMENT_PARSED.value,
        payload={"title": "Test"},
        status=status.value,
        attempts=0,
        created_at=now,
        available_at=available_at or now,
    )
    session.add(row)
    await session.commit()
    return row


async def test_add_get_and_empty_claim(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        await _truncate_outbox(session)
        repo = PostgresOutboxRepository(cast(PostgresSession, session))

        event = OutboxEvent(
            event_id=uuid4(),
            aggregate_type="document",
            aggregate_id=uuid4(),
            event_type=OutboxEventType.DOCUMENT_PARSED,
            payload={"title": "x"},
            created_at=datetime.now(UTC),
        )
        saved = await repo.add(event)
        assert saved.event_id == event.event_id

        loaded = await repo.get(event.event_id)
        assert loaded is not None
        assert loaded.event_type == OutboxEventType.DOCUMENT_PARSED
        assert loaded.status == OutboxEventStatus.PENDING

        claimed = await repo.claim_batch(batch_size=10, worker_id="w1", lease_duration_seconds=30)
        assert len(claimed) == 1

        empty = await repo.claim_batch(batch_size=10, worker_id="w1", lease_duration_seconds=30)
        assert empty == []


async def test_claim_reclaim_and_complete(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        await _truncate_outbox(session)
        repo = PostgresOutboxRepository(cast(PostgresSession, session))
        row = await _insert_event(session)

        claimed = await repo.claim_batch(
            batch_size=1, worker_id="worker-a", lease_duration_seconds=30
        )
        assert len(claimed) == 1
        first = claimed[0]
        assert first.event_id == row.event_id
        stored = await session.get(Outbox, row.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-a"
        assert stored.claim_token is not None

        token = stored.claim_token
        await repo.mark_failed(row.event_id, token, "temporary failure")
        failed = await session.get(Outbox, row.event_id)
        assert failed is not None
        assert failed.status == OutboxEventStatus.PENDING.value
        assert failed.last_error == "temporary failure"
        assert failed.claimed_by is None
        assert failed.claim_token is None
        assert failed.available_at is not None

        immediate = await repo.claim_batch(
            batch_size=1, worker_id="worker-b", lease_duration_seconds=45
        )
        assert immediate == []

        await session.execute(
            update(Outbox)
            .where(Outbox.event_id == row.event_id)
            .values(available_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()

        reclaimed = await repo.claim_batch(
            batch_size=1, worker_id="worker-b", lease_duration_seconds=45
        )
        assert len(reclaimed) == 1
        second = reclaimed[0]
        assert second.event_id == row.event_id
        stored = await session.get(Outbox, row.event_id)
        assert stored is not None
        assert stored.claimed_by == "worker-b"
        assert stored.claim_token is not None
        await repo.mark_completed(row.event_id, stored.claim_token)
        completed = await session.get(Outbox, row.event_id)
        assert completed is not None
        assert completed.status == OutboxEventStatus.COMPLETED.value
        assert completed.completed_at is not None


async def test_claim_token_mismatch_and_backoff(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        await _truncate_outbox(session)
        repo = PostgresOutboxRepository(cast(PostgresSession, session))
        await _insert_event(session)
        other = await _insert_event(session)

        claimed = await repo.claim_batch(
            batch_size=1, worker_id="worker-a", lease_duration_seconds=1
        )
        assert len(claimed) == 1
        claimed_event_id = claimed[0].event_id
        stored = await session.get(Outbox, claimed_event_id)
        assert stored is not None
        token = stored.claim_token or ""

        with pytest.raises(ValueError, match="invalid claim token"):
            await repo.mark_completed(claimed_event_id, "wrong-token")

        with pytest.raises(ValueError, match="invalid claim token"):
            await repo.mark_failed(claimed_event_id, "wrong-token", "nope")

        await repo.mark_failed(claimed_event_id, token, "x" * 600)
        failed = await session.get(Outbox, claimed_event_id)
        assert failed is not None
        assert len(failed.last_error or "") == 512
        assert failed.available_at is not None
        assert failed.available_at > datetime.now(UTC) - timedelta(seconds=1)

        await session.execute(
            update(Outbox)
            .where(Outbox.event_id == claimed_event_id)
            .values(available_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()

        reclaimed = await repo.claim_batch(
            batch_size=1, worker_id="worker-b", lease_duration_seconds=1
        )
        assert len(reclaimed) == 1
        assert reclaimed[0].event_id == claimed_event_id

        # second row is still pending and should not be claimed when batch size is 0
        pending = await session.get(Outbox, other.event_id)
        assert pending is not None
        assert pending.status == OutboxEventStatus.PENDING.value


async def test_claim_lease_recovery_across_sessions(postgres_component: Any) -> None:
    async with _setup_postgres(postgres_component.dsn) as session_factory:
        async with session_factory() as writer_session:
            await _truncate_outbox(writer_session)
            row = await _insert_event(writer_session)

        async with session_factory() as worker_a_session, session_factory() as worker_b_session:
            repo_a = PostgresOutboxRepository(cast(PostgresSession, worker_a_session))
            repo_b = PostgresOutboxRepository(cast(PostgresSession, worker_b_session))

            claimed = await repo_a.claim_batch(
                batch_size=1, worker_id="worker-a", lease_duration_seconds=1
            )
            assert len(claimed) == 1
            token_a = claimed[0].claim_token or ""
            await worker_a_session.commit()

            await worker_b_session.execute(
                update(Outbox)
                .where(Outbox.event_id == row.event_id)
                .values(
                    available_at=datetime.now(UTC) - timedelta(seconds=1),
                    lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
                )
            )
            await worker_b_session.commit()

            reclaimed = await repo_b.claim_batch(
                batch_size=1, worker_id="worker-b", lease_duration_seconds=30
            )
            assert len(reclaimed) == 1
            token_b = reclaimed[0].claim_token or ""
            assert token_b != token_a
            await worker_b_session.commit()

            with pytest.raises(ValueError, match=r"invalid claim token|event claim lost"):
                await repo_a.mark_completed(row.event_id, token_a)

            await repo_b.mark_completed(row.event_id, token_b)
            completed = await worker_b_session.get(Outbox, row.event_id)
            assert completed is not None
            assert completed.status == OutboxEventStatus.COMPLETED.value


async def test_claim_batch_competes_across_sessions(postgres_component: Any) -> None:
    async with _setup_postgres(postgres_component.dsn) as session_factory:
        async with session_factory() as writer_session:
            await _truncate_outbox(writer_session)
            row = await _insert_event(writer_session)

        async with session_factory() as worker_a_session, session_factory() as worker_b_session:
            repo_a = PostgresOutboxRepository(cast(PostgresSession, worker_a_session))
            repo_b = PostgresOutboxRepository(cast(PostgresSession, worker_b_session))
            start = asyncio.Event()

            async def _attempt(
                repo: PostgresOutboxRepository, session: AsyncSession, worker_id: str
            ) -> list[OutboxEvent]:
                await start.wait()
                claimed = await repo.claim_batch(
                    batch_size=1,
                    worker_id=worker_id,
                    lease_duration_seconds=30,
                )
                if claimed:
                    await session.commit()
                else:
                    await session.rollback()
                return claimed

            task_a = asyncio.create_task(_attempt(repo_a, worker_a_session, "worker-a"))
            task_b = asyncio.create_task(_attempt(repo_b, worker_b_session, "worker-b"))
            start.set()
            claimed_a, claimed_b = await asyncio.gather(task_a, task_b)

            total_claimed = len(claimed_a) + len(claimed_b)
            assert total_claimed == 1
            if claimed_a:
                assert claimed_a[0].event_id == row.event_id
                assert claimed_b == []
            else:
                assert claimed_b[0].event_id == row.event_id
                assert claimed_a == []

        async with session_factory() as verify_session:
            stored = await verify_session.get(Outbox, row.event_id)
            assert stored is not None
            assert stored.status == OutboxEventStatus.CLAIMED.value
            assert stored.claimed_by in {"worker-a", "worker-b"}
