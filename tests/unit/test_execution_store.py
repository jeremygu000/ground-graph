"""Unit tests for the PostgreSQL execution repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from groundgraph.domain.execution import (
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
)
from groundgraph.infrastructure.postgres.execution_store import ExecutionRepository
from groundgraph.infrastructure.postgres.models import ExecutionRun as SQLExecutionRun
from groundgraph.infrastructure.postgres.models import ExecutionStep as SQLExecutionStep
from groundgraph.infrastructure.postgres.session import PostgresSession


@dataclass
class _RunRow:
    run_id: UUID
    workflow: str
    status: str
    principal: str
    tenant_id: str
    input: dict[str, Any]
    output: dict[str, Any]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class _StepRow:
    step_id: UUID
    run_id: UUID
    name: str
    status: str
    attempt: int
    depends_on: list[UUID]
    input: dict[str, Any]
    output: dict[str, Any]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class _Result:
    def __init__(self, row: object | None = None, rows: list[object] | None = None) -> None:
        self._row = row
        self._rows = rows or []

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[object]:
        return self._rows

    def scalar_one_or_none(self) -> object | None:
        return self._row

    def scalar_one(self) -> object:
        if self._row is None:
            raise LookupError("no row")
        return self._row


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flushed = 0
        self.executed: list[object] = []
        self.get_map: dict[tuple[object, object], object | None] = {}
        self.execute_result: _Result = _Result()
        self.last_statement: object | None = None

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def flush(self) -> None:
        self.flushed += 1

    async def execute(self, statement: object, parameters: object | None = None) -> _Result:
        self.last_statement = statement
        self.executed.append(statement)
        return self.execute_result

    async def get(self, entity: object, ident: object) -> object | None:
        return self.get_map.get((entity, ident))


@pytest.mark.asyncio
async def test_create_and_get_run_and_step() -> None:
    session = _Session()
    repo = ExecutionRepository(cast(PostgresSession, session))
    run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal="engineering",
        tenant_id="default",
    )
    step = ExecutionStep(
        step_id=uuid4(),
        run_id=run.run_id,
        name="plan",
        status=ExecutionStepStatus.PENDING,
    )

    await repo.create_run(run)
    await repo.create_step(step)
    assert session.flushed == 2


@pytest.mark.asyncio
async def test_get_missing_and_reconstruct_missing() -> None:
    session = _Session()
    repo = ExecutionRepository(cast(PostgresSession, session))
    assert await repo.get_run(uuid4()) is None
    assert await repo.get_step(uuid4()) is None
    with pytest.raises(Exception, match=r"ExecutionRun .* not found"):
        await repo.reconstruct_dag(uuid4())


@pytest.mark.asyncio
async def test_invalid_updates_and_terminal_fields() -> None:
    session = _Session()
    repo = ExecutionRepository(cast(PostgresSession, session))
    run = _RunRow(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING.value,
        principal="engineering",
        tenant_id="default",
        input={},
        output={},
    )
    step = _StepRow(
        step_id=uuid4(),
        run_id=run.run_id,
        name="plan",
        status=ExecutionStepStatus.PENDING.value,
        attempt=1,
        depends_on=[],
        input={},
        output={},
    )
    session.get_map = {
        (SQLExecutionRun, run.run_id): run,
        (SQLExecutionStep, step.step_id): step,
    }
    with pytest.raises(Exception, match=r"illegal execution_run transition"):
        await repo.update_run_status(
            run.run_id,
            ExecutionRunStatus.PENDING,
            ExecutionRunStatus.SUCCEEDED,
        )
    with pytest.raises(Exception, match=r"illegal execution_step transition"):
        await repo.update_step_status(
            step.step_id,
            ExecutionStepStatus.PENDING,
            ExecutionStepStatus.SUCCEEDED,
        )


@pytest.mark.asyncio
async def test_update_run_status_uses_atomic_where_clause() -> None:
    session = _Session()
    repo = ExecutionRepository(cast(PostgresSession, session))
    run_id = uuid4()
    session.execute_result = _Result(
        row=_RunRow(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.RUNNING.value,
            principal="engineering",
            tenant_id="default",
            input={},
            output={},
        )
    )

    await repo.update_run_status(run_id, ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING)

    assert session.last_statement is not None
    assert "status =" in str(session.last_statement)


@pytest.mark.asyncio
async def test_update_step_status_uses_atomic_where_clause() -> None:
    session = _Session()
    repo = ExecutionRepository(cast(PostgresSession, session))
    step_id = uuid4()
    session.execute_result = _Result(
        row=_StepRow(
            step_id=step_id,
            run_id=step_id,
            name="plan",
            status=ExecutionStepStatus.RUNNING.value,
            attempt=1,
            depends_on=[],
            input={},
            output={},
        )
    )

    await repo.update_step_status(step_id, ExecutionStepStatus.PENDING, ExecutionStepStatus.RUNNING)

    assert session.last_statement is not None
    assert "status =" in str(session.last_statement)
