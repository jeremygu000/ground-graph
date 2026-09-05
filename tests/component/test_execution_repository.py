"""Component tests for the PostgreSQL execution repository."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from groundgraph.application.errors import (
    ConcurrencyConflictError,
    InvalidTransitionError,
    NotFoundError,
)
from groundgraph.domain.execution import (
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
)
from groundgraph.infrastructure.postgres.execution_store import ExecutionRepository
from groundgraph.infrastructure.postgres.models import Base as PostgresBase
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


async def test_run_step_persist_and_reconstruct(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))

        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="engineering",
            tenant_id="default",
        )
        await repo.create_run(run)

        step1 = ExecutionStep(
            step_id=uuid4(),
            run_id=run.run_id,
            name="plan",
            status=ExecutionStepStatus.PENDING,
            depends_on=[],
        )
        step2 = ExecutionStep(
            step_id=uuid4(),
            run_id=run.run_id,
            name="answer",
            status=ExecutionStepStatus.PENDING,
            depends_on=[step1.step_id],
        )
        await repo.create_step(step1)
        await repo.create_step(step2)

        restored_run, steps = await repo.reconstruct_dag(run.run_id)
        assert restored_run.workflow == "query"
        assert {s.name for s in steps} == {"plan", "answer"}


async def test_invalid_run_and_step_transitions(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))

        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="engineering",
            tenant_id="default",
        )
        await repo.create_run(run)

        with pytest.raises(InvalidTransitionError, match="illegal execution_run transition"):
            await repo.update_run_status(
                run.run_id,
                ExecutionRunStatus.PENDING,
                ExecutionRunStatus.SUCCEEDED,
            )

        step = ExecutionStep(
            step_id=uuid4(),
            run_id=run.run_id,
            name="plan",
            status=ExecutionStepStatus.PENDING,
            depends_on=[],
        )
        await repo.create_step(step)

        with pytest.raises(InvalidTransitionError, match="illegal execution_step transition"):
            await repo.update_step_status(
                step.step_id,
                ExecutionStepStatus.PENDING,
                ExecutionStepStatus.SUCCEEDED,
            )


async def test_terminal_run_and_step_updates(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="engineering",
            tenant_id="default",
        )
        await repo.create_run(run)
        updated = await repo.update_run_status(
            run.run_id,
            ExecutionRunStatus.PENDING,
            ExecutionRunStatus.RUNNING,
        )
        assert updated.status == ExecutionRunStatus.RUNNING
        finished = await repo.update_run_status(
            run.run_id,
            ExecutionRunStatus.RUNNING,
            ExecutionRunStatus.SUCCEEDED,
        )
        assert finished.status == ExecutionRunStatus.SUCCEEDED
        assert finished.finished_at is not None

        step = ExecutionStep(
            step_id=uuid4(),
            run_id=run.run_id,
            name="answer",
            status=ExecutionStepStatus.PENDING,
            depends_on=[],
        )
        await repo.create_step(step)
        started = await repo.update_step_status(
            step.step_id,
            ExecutionStepStatus.PENDING,
            ExecutionStepStatus.RUNNING,
        )
        assert started.status == ExecutionStepStatus.RUNNING
        succeeded = await repo.update_step_status(
            step.step_id,
            ExecutionStepStatus.RUNNING,
            ExecutionStepStatus.SUCCEEDED,
        )
        assert succeeded.status == ExecutionStepStatus.SUCCEEDED
        assert succeeded.finished_at is not None


async def test_get_missing_run_and_step(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))

        assert await repo.get_run(uuid4()) is None
        assert await repo.get_step(uuid4()) is None


async def test_missing_run_and_step_update_errors(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))

        with pytest.raises(NotFoundError, match=r"ExecutionRun .* not found"):
            await repo.update_run_status(
                uuid4(), ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING
            )

        with pytest.raises(NotFoundError, match=r"ExecutionStep .* not found"):
            await repo.update_step_status(
                uuid4(), ExecutionStepStatus.PENDING, ExecutionStepStatus.RUNNING
            )


async def test_get_steps_for_run_returns_empty(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="engineering",
            tenant_id="default",
        )
        await repo.create_run(run)
        assert await repo.get_steps_for_run(run.run_id) == []


async def test_run_update_terminal_fields(postgres_component: Any) -> None:
    async with (
        _setup_postgres(postgres_component.dsn) as session_factory,
        session_factory() as session,
    ):
        repo = ExecutionRepository(cast(PostgresSession, session))
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="engineering",
            tenant_id="default",
        )
        await repo.create_run(run)

        running = await repo.update_run_status(
            run.run_id, ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING
        )
        assert running.status == ExecutionRunStatus.RUNNING
        finished = await repo.update_run_status(
            run.run_id,
            ExecutionRunStatus.RUNNING,
            ExecutionRunStatus.FAILED,
            error_code="oops",
            error_message="boom",
        )
        assert finished.status == ExecutionRunStatus.FAILED
        assert finished.finished_at is not None
        assert finished.error_code == "oops"
        assert finished.error_message == "boom"


async def test_run_update_conflicts_across_sessions(postgres_component: Any) -> None:
    async with _setup_postgres(postgres_component.dsn) as session_factory:
        async with session_factory() as writer_session:
            writer_repo = ExecutionRepository(cast(PostgresSession, writer_session))
            run = ExecutionRun(
                run_id=uuid4(),
                workflow="query",
                status=ExecutionRunStatus.PENDING,
                principal="engineering",
                tenant_id="default",
            )
            await writer_repo.create_run(run)
            await writer_session.commit()

        async with session_factory() as winner_session, session_factory() as loser_session:
            winner_repo = ExecutionRepository(cast(PostgresSession, winner_session))
            loser_repo = ExecutionRepository(cast(PostgresSession, loser_session))
            start = asyncio.Event()

            async def _attempt(
                repo: ExecutionRepository, session: AsyncSession
            ) -> tuple[str, ExecutionRun | ConcurrencyConflictError]:
                await start.wait()
                try:
                    updated = await repo.update_run_status(
                        run.run_id,
                        ExecutionRunStatus.PENDING,
                        ExecutionRunStatus.RUNNING,
                    )
                    await session.commit()
                except ConcurrencyConflictError as exc:
                    await session.rollback()
                    return ("conflict", exc)
                else:
                    return ("ok", updated)

            winner_task = asyncio.create_task(_attempt(winner_repo, winner_session))
            loser_task = asyncio.create_task(_attempt(loser_repo, loser_session))
            start.set()
            results = await asyncio.gather(winner_task, loser_task)

            ok = [result for result in results if result[0] == "ok"]
            conflicts = [result for result in results if result[0] == "conflict"]
            assert len(ok) == 1
            assert len(conflicts) == 1
            winner = ok[0][1]
            assert isinstance(winner, ExecutionRun)
            assert winner.status == ExecutionRunStatus.RUNNING

        async with session_factory() as verify_session:
            verify_repo = ExecutionRepository(cast(PostgresSession, verify_session))
            loaded = await verify_repo.get_run(run.run_id)
            assert loaded is not None
            assert loaded.status == ExecutionRunStatus.RUNNING


async def test_step_update_conflicts_across_sessions(postgres_component: Any) -> None:
    async with _setup_postgres(postgres_component.dsn) as session_factory:
        async with session_factory() as writer_session:
            writer_repo = ExecutionRepository(cast(PostgresSession, writer_session))
            run = ExecutionRun(
                run_id=uuid4(),
                workflow="query",
                status=ExecutionRunStatus.PENDING,
                principal="engineering",
                tenant_id="default",
            )
            await writer_repo.create_run(run)
            step = ExecutionStep(
                step_id=uuid4(),
                run_id=run.run_id,
                name="plan",
                status=ExecutionStepStatus.PENDING,
                depends_on=[],
            )
            await writer_repo.create_step(step)
            await writer_session.commit()

        async with session_factory() as winner_session, session_factory() as loser_session:
            winner_repo = ExecutionRepository(cast(PostgresSession, winner_session))
            loser_repo = ExecutionRepository(cast(PostgresSession, loser_session))
            start = asyncio.Event()

            async def _attempt(
                repo: ExecutionRepository, session: AsyncSession
            ) -> tuple[str, ExecutionStep | ConcurrencyConflictError]:
                await start.wait()
                try:
                    updated = await repo.update_step_status(
                        step.step_id,
                        ExecutionStepStatus.PENDING,
                        ExecutionStepStatus.RUNNING,
                    )
                    await session.commit()
                except ConcurrencyConflictError as exc:
                    await session.rollback()
                    return ("conflict", exc)
                else:
                    return ("ok", updated)

            winner_task = asyncio.create_task(_attempt(winner_repo, winner_session))
            loser_task = asyncio.create_task(_attempt(loser_repo, loser_session))
            start.set()
            results = await asyncio.gather(winner_task, loser_task)

            ok = [result for result in results if result[0] == "ok"]
            conflicts = [result for result in results if result[0] == "conflict"]
            assert len(ok) == 1
            assert len(conflicts) == 1
            winner = ok[0][1]
            assert isinstance(winner, ExecutionStep)
            assert winner.status == ExecutionStepStatus.RUNNING

        async with session_factory() as verify_session:
            verify_repo = ExecutionRepository(cast(PostgresSession, verify_session))
            loaded = await verify_repo.get_step(step.step_id)
            assert loaded is not None
            assert loaded.status == ExecutionStepStatus.RUNNING
