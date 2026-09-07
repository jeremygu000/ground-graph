"""Unit tests for execution API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from groundgraph.api.dependencies import Identity
from groundgraph.api.execution import (
    ExecutionRunResponse,
    ReplayResponse,
    get_execution,
    replay_execution,
)
from groundgraph.domain.execution import ExecutionRun, ExecutionRunStatus
from groundgraph.domain.retrieval import QueryResponse


class _FakeExecutionRepo:
    def __init__(self) -> None:
        self.runs: dict[UUID, ExecutionRun] = {}

    async def create_run(self, run: ExecutionRun) -> ExecutionRun:
        self.runs[run.run_id] = run
        return run

    async def get_run(self, run_id: UUID) -> ExecutionRun | None:
        return self.runs.get(run_id)

    async def update_run_status(
        self,
        run_id: UUID,
        expected_status: ExecutionRunStatus,
        new_status: ExecutionRunStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionRun:
        return self.runs[run_id]


class _FakeWorkflow:
    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.ainvoke_calls: list[dict[str, object]] = []

    async def ainvoke(self, **kwargs: Any) -> object:
        self.ainvoke_calls.append(kwargs)
        if self.raises:
            raise self.raises
        return QueryResponse(
            execution_run_id=uuid4(),
            answer="Test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )


@pytest.mark.asyncio
async def test_replay_execution_success() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    original = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.SUCCEEDED,
        principal="user1",
        tenant_id="tenant-a",
        input={"question": "What is Postgres?"},
        output={"answer": "A database."},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = original
    workflow = _FakeWorkflow()
    identity = Identity(tenant_id="tenant-a", principal="user1")

    result = await replay_execution(
        run_id=run_id,
        repo=repo,
        workflow=cast(Any, workflow),
        identity=identity,
    )
    assert result.original_run_id == run_id
    assert result.new_run_id != run_id
    assert result.status == "succeeded"
    assert len(workflow.ainvoke_calls) == 1


@pytest.mark.asyncio
async def test_replay_execution_not_found() -> None:
    repo = _FakeExecutionRepo()
    workflow = _FakeWorkflow()
    identity = Identity(tenant_id="tenant-a", principal="user1")

    with pytest.raises(HTTPException) as exc_info:
        await replay_execution(
            run_id=uuid4(), repo=repo, workflow=cast(Any, workflow), identity=identity
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_replay_execution_cross_tenant_raises_404() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    original = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.SUCCEEDED,
        principal="user1",
        tenant_id="tenant-b",
        input={"question": "What is Postgres?"},
        output={},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = original
    workflow = _FakeWorkflow()
    identity = Identity(tenant_id="tenant-a", principal="user1")

    with pytest.raises(HTTPException) as exc_info:
        await replay_execution(
            run_id=run_id, repo=repo, workflow=cast(Any, workflow), identity=identity
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_replay_execution_non_succeeded_raises_409() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    original = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.FAILED,
        principal="user1",
        tenant_id="tenant-a",
        input={"question": "What is Postgres?"},
        output={},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = original
    workflow = _FakeWorkflow()
    identity = Identity(tenant_id="tenant-a", principal="user1")

    with pytest.raises(HTTPException) as exc_info:
        await replay_execution(
            run_id=run_id, repo=repo, workflow=cast(Any, workflow), identity=identity
        )
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_replay_execution_no_question_raises_422() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    original = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.SUCCEEDED,
        principal="user1",
        tenant_id="tenant-a",
        input={},
        output={},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = original
    workflow = _FakeWorkflow()
    identity = Identity(tenant_id="tenant-a", principal="user1")

    with pytest.raises(HTTPException) as exc_info:
        await replay_execution(
            run_id=run_id, repo=repo, workflow=cast(Any, workflow), identity=identity
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_execution_returns_200() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    run = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.SUCCEEDED,
        principal="user1",
        tenant_id="tenant-a",
        input={"question": "What is Postgres?"},
        output={"answer": "A database."},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = run
    identity = Identity(tenant_id="tenant-a", principal="user1")

    result = await get_execution(
        run_id=run_id,
        repo=repo,
        identity=identity,
    )
    assert isinstance(result, ExecutionRunResponse)
    assert result.run_id == run_id
    assert result.status == "succeeded"


@pytest.mark.asyncio
async def test_get_execution_cross_tenant_raises_404() -> None:
    run_id = uuid4()
    now = datetime.now(UTC)
    run = ExecutionRun(
        run_id=run_id,
        workflow="query",
        status=ExecutionRunStatus.SUCCEEDED,
        principal="user1",
        tenant_id="tenant-b",
        input={},
        output={},
        started_at=now,
        finished_at=now,
    )
    repo = _FakeExecutionRepo()
    repo.runs[run_id] = run
    identity = Identity(tenant_id="tenant-a", principal="user1")

    with pytest.raises(HTTPException) as exc_info:
        await get_execution(run_id=run_id, repo=repo, identity=identity)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_execution_not_found() -> None:
    repo = _FakeExecutionRepo()
    identity = Identity(tenant_id="tenant-a", principal="user1")
    with pytest.raises(HTTPException) as exc_info:
        await get_execution(run_id=uuid4(), repo=repo, identity=identity)
    assert exc_info.value.status_code == 404


class TestExecutionRunResponse:
    def test_from_domain(self) -> None:
        run_id = uuid4()
        started = datetime.now(UTC)
        finished = datetime.now(UTC)
        run = ExecutionRun(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.SUCCEEDED,
            principal="user1",
            tenant_id="tenant-a",
            input={"question": "What is Postgres?"},
            output={"answer": "A database."},
            started_at=started,
            finished_at=finished,
        )

        resp = ExecutionRunResponse.from_domain(run)
        assert resp.run_id == run_id
        assert resp.workflow == "query"
        assert resp.status == "succeeded"
        assert resp.principal == "user1"
        assert resp.tenant_id == "tenant-a"
        assert resp.input == {"question": "What is Postgres?"}
        assert resp.output == {"answer": "A database."}
        assert resp.started_at == started.isoformat()
        assert resp.finished_at == finished.isoformat()
        assert resp.error_code is None
        assert resp.error_message is None

    def test_from_domain_with_error(self) -> None:
        run_id = uuid4()
        run = ExecutionRun(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.FAILED,
            principal="user1",
            tenant_id="tenant-a",
            input={"question": "What is Postgres?"},
            output={},
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            error_code="WORKFLOW_FAILED",
            error_message="Something went wrong",
        )

        resp = ExecutionRunResponse.from_domain(run)
        assert resp.status == "failed"
        assert resp.error_code == "WORKFLOW_FAILED"
        assert resp.error_message == "Something went wrong"

    def test_from_domain_pending_run(self) -> None:
        run_id = uuid4()
        started = datetime.now(UTC)
        run = ExecutionRun(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="user1",
            tenant_id="tenant-a",
            input={"question": "What is Postgres?"},
            output={},
            started_at=started,
        )

        resp = ExecutionRunResponse.from_domain(run)
        assert resp.status == "pending"
        assert resp.started_at == started.isoformat()
        assert resp.finished_at is None

    def test_from_domain_running_status(self) -> None:
        run_id = uuid4()
        started = datetime.now(UTC)
        run = ExecutionRun(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.RUNNING,
            principal="user1",
            tenant_id="tenant-a",
            input={"question": "What is Postgres?"},
            output={},
            started_at=started,
        )
        resp = ExecutionRunResponse.from_domain(run)
        assert resp.status == "running"

    def test_from_domain_cancelled_status(self) -> None:
        run_id = uuid4()
        run = ExecutionRun(
            run_id=run_id,
            workflow="query",
            status=ExecutionRunStatus.CANCELLED,
            principal="user1",
            tenant_id="tenant-a",
            input={},
            output={},
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        resp = ExecutionRunResponse.from_domain(run)
        assert resp.status == "cancelled"


class TestReplayResponse:
    def test_fields(self) -> None:
        new_id = uuid4()
        orig_id = uuid4()
        resp = ReplayResponse(
            new_run_id=new_id,
            original_run_id=orig_id,
            status="succeeded",
            execution_run_id=new_id,
        )
        assert resp.new_run_id == new_id
        assert resp.original_run_id == orig_id
        assert resp.status == "succeeded"
        assert resp.execution_run_id == new_id


class TestExecutionRunStateMachine:
    def test_pending_to_succeeded_transition(self) -> None:
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.PENDING,
            principal="user1",
            tenant_id="tenant-a",
            input={},
            output={},
            started_at=datetime.now(UTC),
        )
        assert run.status == ExecutionRunStatus.PENDING

    def test_terminal_status_requires_finished_at(self) -> None:
        with pytest.raises(ValueError, match="terminal status"):
            ExecutionRun(
                run_id=uuid4(),
                workflow="query",
                status=ExecutionRunStatus.SUCCEEDED,
                principal="user1",
                tenant_id="tenant-a",
                input={},
                output={},
                started_at=datetime.now(UTC),
            )

    def test_terminal_status_with_finished_at(self) -> None:
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.SUCCEEDED,
            principal="user1",
            tenant_id="tenant-a",
            input={},
            output={},
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        assert run.status == ExecutionRunStatus.SUCCEEDED

    def test_cancelled_status(self) -> None:
        run = ExecutionRun(
            run_id=uuid4(),
            workflow="query",
            status=ExecutionRunStatus.CANCELLED,
            principal="user1",
            tenant_id="tenant-a",
            input={},
            output={},
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        assert run.status == ExecutionRunStatus.CANCELLED
