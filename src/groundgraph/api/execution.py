"""Execution inspection and replay endpoints (M7)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from groundgraph.api.dependencies import (
    Identity,
    get_execution_repo_with_session,
    get_identity,
    get_query_workflow,
)
from groundgraph.domain.execution import ExecutionRun, ExecutionRunStatus
from groundgraph.workflows.query_graph import QueryWorkflow

router = APIRouter(prefix="/v1", tags=["v1-execution"])


class ExecutionRunResponse(BaseModel):
    run_id: UUID
    workflow: str
    status: str
    principal: str
    tenant_id: str
    input: dict[str, object] = Field(default_factory=dict)
    output: dict[str, object] = Field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_domain(cls, run: ExecutionRun) -> ExecutionRunResponse:
        return cls(
            run_id=run.run_id,
            workflow=run.workflow,
            status=run.status.value,
            principal=run.principal,
            tenant_id=run.tenant_id,
            input=run.input,
            output=run.output,
            started_at=run.started_at.isoformat() if run.started_at else None,
            finished_at=run.finished_at.isoformat() if run.finished_at else None,
            error_code=run.error_code,
            error_message=run.error_message,
        )


class ReplayResponse(BaseModel):
    new_run_id: UUID
    original_run_id: UUID
    status: str
    execution_run_id: UUID


@router.get(
    "/executions/{run_id}",
    response_model=ExecutionRunResponse,
    name="get-execution",
)
async def get_execution(
    run_id: UUID,
    repo: Annotated[object, Depends(get_execution_repo_with_session)],
    identity: Annotated[Identity, Depends(get_identity)],
) -> ExecutionRunResponse:
    """Retrieve a previously executed run by ID.

    The run must belong to the requesting tenant. Tenants cannot inspect
    each other's execution runs.
    """
    result = await repo.get_run(run_id)  # type: ignore[attr-defined]
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        )
    if result.tenant_id != identity.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        )
    return ExecutionRunResponse.from_domain(result)


@router.post(
    "/executions/{run_id}/replay",
    response_model=ReplayResponse,
    name="replay-execution",
)
async def replay_execution(
    run_id: UUID,
    repo: Annotated[object, Depends(get_execution_repo_with_session)],
    workflow: Annotated[QueryWorkflow, Depends(get_query_workflow)],
    identity: Annotated[Identity, Depends(get_identity)],
) -> ReplayResponse:
    """Replay a previously executed query using the same input parameters.

    A new execution run is created with a fresh run_id. The original run
    is preserved and not modified. Versioned inputs (index_version, etc.)
    are taken from the original run's stored input, not recomputed.

    Only runs with status='succeeded' can be replayed.
    """
    original = await repo.get_run(run_id)  # type: ignore[attr-defined]
    if original is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        )
    if original.tenant_id != identity.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        )
    if original.status != ExecutionRunStatus.SUCCEEDED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot replay run in status {original.status.value}. "
                "Only succeeded runs can be replayed."
            ),
        )

    question = original.input.get("question")
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Original run has no question in input; replay requires a query-type run.",
        )

    new_run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal=identity.principal,
        tenant_id=identity.tenant_id,
        input=original.input,
        output={},
        started_at=datetime.now(UTC),
    )
    await repo.create_run(new_run)  # type: ignore[attr-defined]

    try:
        await workflow.ainvoke(
            question=question,
            principal=identity.principal,
            tenant_id=identity.tenant_id,
            index_name=original.input.get("index_name"),
        )
        await repo.update_run_status(  # type: ignore[attr-defined]
            run_id=new_run.run_id,
            expected_status=ExecutionRunStatus.PENDING,
            new_status=ExecutionRunStatus.SUCCEEDED,
        )
    except Exception as exc:
        await repo.update_run_status(  # type: ignore[attr-defined]
            run_id=new_run.run_id,
            expected_status=ExecutionRunStatus.PENDING,
            new_status=ExecutionRunStatus.FAILED,
            error_code="REPLAY_FAILED",
            error_message=str(exc),
        )
        raise

    return ReplayResponse(
        new_run_id=new_run.run_id,
        original_run_id=run_id,
        status="succeeded",
        execution_run_id=new_run.run_id,
    )
