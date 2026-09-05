"""Unit tests for the PostgreSQL execution repository."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

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


@dataclass
class _RunRow:
    run_id: UUID
    workflow: str
    status: str
    principal: str
    tenant_id: str
    input: dict[str, object]
    output: dict[str, object]
    started_at: object = None
    finished_at: object = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class _StepRow:
    step_id: UUID
    run_id: UUID
    name: str
    status: str
    attempt: int
    depends_on: list[object]
    input: dict[str, object]
    output: dict[str, object]
    started_at: object = None
    finished_at: object = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: object = None


class _Result:
    def __init__(self, row: object | None = None, rows: list[object] | None = None) -> None:
        self._row = row
        self._rows = rows or []

    def scalar_one_or_none(self) -> object | None:
        return self._row

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[object]:
        return self._rows


class _ExecuteFactory:
    def __init__(self, row: object | None = None, rows: Sequence[object] | None = None) -> None:
        self.row = row
        self.rows = list(rows or [])

    async def __call__(self, _statement: object, _parameters: object | None = None) -> _Result:
        return _Result(row=self.row, rows=self.rows)


class _Session:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flushed = 0
        self.executed: list[object] = []
        self.get_map: dict[tuple[object, object], object | None] = {}
        self.execute_factory = _ExecuteFactory()

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object, parameters: object | None = None) -> _Result:
        self.executed.append(statement)
        return await self.execute_factory(statement, parameters)

    async def get(self, entity: object, ident: object) -> object | None:
        return self.get_map.get((entity, ident))

    async def flush(self) -> None:
        self.flushed += 1

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_execution_repository_create_and_get_run() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal="engineering",
        tenant_id="default",
        input={"q": "hello"},
    )
    created = await repo.create_run(run)
    assert created == run
    assert session.flushed == 1
    assert len(session.added) == 1

    stored = _RunRow(
        run_id=run.run_id,
        workflow=run.workflow,
        status=run.status.value,
        principal=run.principal,
        tenant_id=run.tenant_id,
        input=run.input,
        output=run.output,
        started_at=datetime.now(UTC),
    )
    session.execute_factory = _ExecuteFactory(row=stored)
    loaded = await repo.get_run(run.run_id)
    assert loaded is not None
    assert loaded.workflow == "query"


@pytest.mark.asyncio
async def test_execution_repository_create_step_and_list_steps() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    run_id = uuid4()
    step1 = ExecutionStep(
        step_id=uuid4(),
        run_id=run_id,
        name="plan",
        status=ExecutionStepStatus.PENDING,
        depends_on=[],
    )
    step2 = ExecutionStep(
        step_id=uuid4(),
        run_id=run_id,
        name="answer",
        status=ExecutionStepStatus.PENDING,
        depends_on=[step1.step_id],
    )
    await repo.create_step(step1)
    await repo.create_step(step2)
    assert session.flushed == 2
    assert len(session.added) == 2

    rows = [
        _StepRow(
            step_id=step1.step_id,
            run_id=run_id,
            name=step1.name,
            status=step1.status.value,
            attempt=1,
            depends_on=[],
            input={},
            output={},
            created_at=datetime.now(UTC),
        ),
        _StepRow(
            step_id=step2.step_id,
            run_id=run_id,
            name=step2.name,
            status=step2.status.value,
            attempt=1,
            depends_on=[step1.step_id],
            input={},
            output={},
            created_at=datetime.now(UTC),
        ),
    ]
    session.execute_factory = _ExecuteFactory(rows=rows)
    steps = await repo.get_steps_for_run(run_id)
    assert [step.name for step in steps] == ["plan", "answer"]


@pytest.mark.asyncio
async def test_execution_repository_get_step_and_reconstruct_dag() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    run_id = uuid4()
    step_id = uuid4()
    run_row = _RunRow(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.RUNNING.value,
        principal="engineering",
        tenant_id="default",
        input={},
        output={},
    )
    step_row = _StepRow(
        step_id=step_id,
        run_id=run_id,
        name="plan",
        status=ExecutionStepStatus.RUNNING.value,
        attempt=1,
        depends_on=[],
        input={},
        output={},
        created_at=datetime.now(UTC),
    )

    session.execute_factory = _ExecuteFactory(row=run_row)
    loaded_run = await repo.get_run(run_id)
    assert loaded_run is not None

    session.execute_factory = _ExecuteFactory(row=step_row)
    loaded_step = await repo.get_step(step_id)
    assert loaded_step is not None
    assert loaded_step.name == "plan"

    session.execute_factory = _ExecuteFactory(row=run_row, rows=[step_row])
    reconstructed_run, reconstructed_steps = await repo.reconstruct_dag(run_id)
    assert reconstructed_run.run_id == run_id
    assert len(reconstructed_steps) == 1


@pytest.mark.asyncio
async def test_execution_repository_invalid_updates_and_reconstruct_missing_run() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    with pytest.raises(NotFoundError, match=r"ExecutionRun .* not found"):
        await repo.update_run_status(
            uuid4(), ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING
        )

    with pytest.raises(NotFoundError, match=r"ExecutionStep .* not found"):
        await repo.update_step_status(
            uuid4(), ExecutionStepStatus.PENDING, ExecutionStepStatus.RUNNING
        )

    with pytest.raises(ValueError, match=r"ExecutionRun .* not found"):
        await repo.reconstruct_dag(uuid4())


@pytest.mark.asyncio
async def test_execution_repository_rejects_mutated_json_input() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal="engineering",
        tenant_id="default",
        input={"ok": {"nested": "value"}},
    )
    cast(dict[str, object], run.input["ok"])["blob"] = b"raw"

    with pytest.raises(ValueError, match="JsonValue"):
        await repo.create_run(run)

    assert session.added == []
    assert session.flushed == 0


@pytest.mark.asyncio
async def test_execution_repository_terminal_run_and_step_updates() -> None:
    session = _Session()
    run_id = uuid4()
    step_id = uuid4()
    repo = ExecutionRepository(session)

    session.execute_factory = _ExecuteFactory(
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

    session.execute_factory = _ExecuteFactory(
        row=_RunRow(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.SUCCEEDED.value,
            principal="engineering",
            tenant_id="default",
            input={},
            output={},
            finished_at=datetime.now(UTC),
        )
    )
    updated_run = await repo.update_run_status(
        run_id, ExecutionRunStatus.RUNNING, ExecutionRunStatus.SUCCEEDED
    )
    assert updated_run.status == ExecutionRunStatus.SUCCEEDED
    assert updated_run.finished_at is not None

    session.execute_factory = _ExecuteFactory(
        row=_StepRow(
            step_id=step_id,
            run_id=run_id,
            name="plan",
            status=ExecutionStepStatus.RUNNING.value,
            attempt=1,
            depends_on=[],
            input={},
            output={},
            created_at=datetime.now(UTC),
        )
    )
    await repo.update_step_status(step_id, ExecutionStepStatus.PENDING, ExecutionStepStatus.RUNNING)

    session.execute_factory = _ExecuteFactory(
        row=_StepRow(
            step_id=step_id,
            run_id=run_id,
            name="plan",
            status=ExecutionStepStatus.SUCCEEDED.value,
            attempt=1,
            depends_on=[],
            input={},
            output={},
            finished_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
        )
    )
    updated_step = await repo.update_step_status(
        step_id, ExecutionStepStatus.RUNNING, ExecutionStepStatus.SUCCEEDED
    )
    assert updated_step.status == ExecutionStepStatus.SUCCEEDED
    assert updated_step.finished_at is not None


@pytest.mark.asyncio
async def test_execution_repository_invalid_transition_rejected_before_write() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    with pytest.raises(InvalidTransitionError):
        await repo.update_run_status(
            uuid4(), ExecutionRunStatus.PENDING, ExecutionRunStatus.SUCCEEDED
        )

    with pytest.raises(InvalidTransitionError):
        await repo.update_step_status(
            uuid4(), ExecutionStepStatus.PENDING, ExecutionStepStatus.SUCCEEDED
        )


@pytest.mark.asyncio
async def test_execution_repository_conflict_when_expected_status_mismatches() -> None:
    session = _Session()
    repo = ExecutionRepository(session)

    run_id = uuid4()
    session.get_map = {
        (
            __import__(
                "groundgraph.infrastructure.postgres.models", fromlist=["ExecutionRun"]
            ).ExecutionRun,
            run_id,
        ): _RunRow(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.RUNNING.value,
            principal="engineering",
            tenant_id="default",
            input={},
            output={},
        )
    }
    session.execute_factory = _ExecuteFactory(row=None)

    with pytest.raises(ConcurrencyConflictError):
        await repo.update_run_status(run_id, ExecutionRunStatus.PENDING, ExecutionRunStatus.RUNNING)
