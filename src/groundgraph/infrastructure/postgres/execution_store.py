"""PostgreSQL execution store (plan.md §2.2 ExecutionRepository port).

Implements the ExecutionRepository port using SQLAlchemy 2.x async.
State machine transitions are enforced via assert_run_transition().
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.domain.execution import (
    ALLOWED_STEP_TRANSITIONS,
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
    assert_run_transition,
)
from groundgraph.infrastructure.postgres.models import ExecutionRun as SQLExecutionRun
from groundgraph.infrastructure.postgres.models import ExecutionStep as SQLExecutionStep


class ExecutionRepository:
    """PostgreSQL adapter for execution run/step persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(self, run: ExecutionRun) -> ExecutionRun:
        sql_run = SQLExecutionRun(
            run_id=run.run_id,
            workflow=run.workflow,
            status=run.status.value,
            principal=run.principal,
            tenant_id=run.tenant_id,
            input=run.input,
            output=run.output,
            started_at=run.started_at,
            finished_at=run.finished_at,
            error_code=run.error_code,
            error_message=run.error_message,
        )
        self._session.add(sql_run)
        await self._session.commit()
        return run

    async def get_run(self, run_id: UUID) -> ExecutionRun | None:
        result = await self._session.execute(
            select(SQLExecutionRun).where(SQLExecutionRun.run_id == run_id)
        )
        sql_run = result.scalar_one_or_none()
        if not sql_run:
            return None
        return self._sql_run_to_domain(sql_run)

    async def update_run_status(
        self,
        run_id: UUID,
        new_status: ExecutionRunStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionRun:
        sql_run = await self._session.get(SQLExecutionRun, run_id)
        if not sql_run:
            raise ValueError(f"ExecutionRun {run_id} not found")

        current = ExecutionRunStatus(sql_run.status)
        assert_run_transition(current, new_status)

        sql_run.status = new_status.value
        if new_status in (
            ExecutionRunStatus.SUCCEEDED,
            ExecutionRunStatus.FAILED,
            ExecutionRunStatus.CANCELLED,
        ):
            sql_run.finished_at = datetime.now(UTC)
        if error_code is not None:
            sql_run.error_code = error_code
        if error_message is not None:
            sql_run.error_message = error_message

        await self._session.commit()
        return self._sql_run_to_domain(sql_run)

    async def create_step(self, step: ExecutionStep) -> ExecutionStep:
        sql_step = SQLExecutionStep(
            step_id=step.step_id,
            run_id=step.run_id,
            name=step.name,
            status=step.status.value,
            attempt=step.attempt,
            depends_on=step.depends_on,
            input=step.input,
            output=step.output,
            started_at=step.started_at,
            finished_at=step.finished_at,
            error_code=step.error_code,
            error_message=step.error_message,
        )
        self._session.add(sql_step)
        await self._session.commit()
        return step

    async def get_step(self, step_id: UUID) -> ExecutionStep | None:
        result = await self._session.execute(
            select(SQLExecutionStep).where(SQLExecutionStep.step_id == step_id)
        )
        sql_step = result.scalar_one_or_none()
        if not sql_step:
            return None
        return self._sql_step_to_domain(sql_step)

    async def get_steps_for_run(self, run_id: UUID) -> list[ExecutionStep]:
        result = await self._session.execute(
            select(SQLExecutionStep)
            .where(SQLExecutionStep.run_id == run_id)
            .order_by(SQLExecutionStep.created_at)
        )
        sql_steps = result.scalars().all()
        return [self._sql_step_to_domain(s) for s in sql_steps]

    async def update_step_status(
        self,
        step_id: UUID,
        new_status: ExecutionStepStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionStep:
        sql_step = await self._session.get(SQLExecutionStep, step_id)
        if not sql_step:
            raise ValueError(f"ExecutionStep {step_id} not found")

        current = ExecutionStepStatus(sql_step.status)
        if new_status not in ALLOWED_STEP_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"illegal execution_step transition: {current.value} -> {new_status.value}"
            )

        sql_step.status = new_status.value
        if new_status in (
            ExecutionStepStatus.SUCCEEDED,
            ExecutionStepStatus.FAILED,
            ExecutionStepStatus.SKIPPED,
        ):
            sql_step.finished_at = datetime.now(UTC)
        if error_code is not None:
            sql_step.error_code = error_code
        if error_message is not None:
            sql_step.error_message = error_message

        await self._session.commit()
        return self._sql_step_to_domain(sql_step)

    async def reconstruct_dag(self, run_id: UUID) -> tuple[ExecutionRun, list[ExecutionStep]]:
        run = await self.get_run(run_id)
        if not run:
            raise ValueError(f"ExecutionRun {run_id} not found")
        steps = await self.get_steps_for_run(run_id)
        return run, steps

    def _sql_run_to_domain(self, sql: SQLExecutionRun) -> ExecutionRun:
        return ExecutionRun(
            run_id=sql.run_id,
            workflow=sql.workflow,
            status=ExecutionRunStatus(sql.status),
            principal=sql.principal,
            tenant_id=sql.tenant_id,
            input=sql.input,
            output=sql.output,
            started_at=sql.started_at,
            finished_at=sql.finished_at,
            error_code=sql.error_code,
            error_message=sql.error_message,
        )

    def _sql_step_to_domain(self, sql: SQLExecutionStep) -> ExecutionStep:
        return ExecutionStep(
            step_id=sql.step_id,
            run_id=sql.run_id,
            name=sql.name,
            status=ExecutionStepStatus(sql.status),
            attempt=sql.attempt,
            depends_on=sql.depends_on,
            input=sql.input,
            output=sql.output,
            started_at=sql.started_at,
            finished_at=sql.finished_at,
            error_code=sql.error_code,
            error_message=sql.error_message,
        )
