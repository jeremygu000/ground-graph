"""PostgreSQL execution store (plan.md §2.2 ExecutionRepository port).

Implements the ExecutionRepository port using SQLAlchemy 2.x async.
State machine transitions are enforced via assert_run_transition().
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update

from groundgraph.application.errors import (
    ConcurrencyConflictError,
    InvalidTransitionError,
    NotFoundError,
)
from groundgraph.domain.execution import (
    ALLOWED_RUN_TRANSITIONS,
    ALLOWED_STEP_TRANSITIONS,
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
)
from groundgraph.infrastructure.postgres.models import ExecutionRun as SQLExecutionRun
from groundgraph.infrastructure.postgres.models import ExecutionStep as SQLExecutionStep
from groundgraph.infrastructure.postgres.session import PostgresSession


class ExecutionRepository:
    """PostgreSQL adapter for execution run/step persistence."""

    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def create_run(self, run: ExecutionRun) -> ExecutionRun:
        # The caller owns the transaction boundary (UoW); this repository
        # only flushes changes so the session can be committed or rolled back
        # as one unit with related document/outbox writes.
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
        await self._session.flush()
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
        expected_status: ExecutionRunStatus,
        new_status: ExecutionRunStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionRun:
        if new_status not in ALLOWED_RUN_TRANSITIONS[expected_status]:
            raise InvalidTransitionError(
                f"illegal execution_run transition: {expected_status.value} -> {new_status.value}"
            )

        update_values: dict[str, object] = {"status": new_status.value}
        if new_status in (
            ExecutionRunStatus.SUCCEEDED,
            ExecutionRunStatus.FAILED,
            ExecutionRunStatus.CANCELLED,
        ):
            update_values["finished_at"] = datetime.now(UTC)
        if error_code is not None:
            update_values["error_code"] = error_code
        if error_message is not None:
            update_values["error_message"] = error_message

        result = await self._session.execute(
            update(SQLExecutionRun)
            .where(SQLExecutionRun.run_id == run_id)
            .where(SQLExecutionRun.status == expected_status.value)
            .values(**update_values)
            .returning(SQLExecutionRun)
        )
        sql_run = result.scalar_one_or_none()
        if sql_run is None:
            existing = await self._session.get(SQLExecutionRun, run_id)
            if existing is None:
                raise NotFoundError(f"ExecutionRun {run_id} not found")
            if ExecutionRunStatus(existing.status) != expected_status:
                raise ConcurrencyConflictError(
                    f"ExecutionRun {run_id} status changed from {expected_status.value}"
                )
            raise ConcurrencyConflictError(f"ExecutionRun {run_id} update lost race")
        await self._session.flush()
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
        await self._session.flush()
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
        expected_status: ExecutionStepStatus,
        new_status: ExecutionStepStatus,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ExecutionStep:
        if new_status not in ALLOWED_STEP_TRANSITIONS[expected_status]:
            raise InvalidTransitionError(
                f"illegal execution_step transition: {expected_status.value} -> {new_status.value}"
            )

        update_values: dict[str, object] = {"status": new_status.value}
        if new_status in (
            ExecutionStepStatus.SUCCEEDED,
            ExecutionStepStatus.FAILED,
            ExecutionStepStatus.SKIPPED,
        ):
            update_values["finished_at"] = datetime.now(UTC)
        if error_code is not None:
            update_values["error_code"] = error_code
        if error_message is not None:
            update_values["error_message"] = error_message

        result = await self._session.execute(
            update(SQLExecutionStep)
            .where(SQLExecutionStep.step_id == step_id)
            .where(SQLExecutionStep.status == expected_status.value)
            .values(**update_values)
            .returning(SQLExecutionStep)
        )
        sql_step = result.scalar_one_or_none()
        if sql_step is None:
            existing = await self._session.get(SQLExecutionStep, step_id)
            if existing is None:
                raise NotFoundError(f"ExecutionStep {step_id} not found")
            if ExecutionStepStatus(existing.status) != expected_status:
                raise ConcurrencyConflictError(
                    f"ExecutionStep {step_id} status changed from {expected_status.value}"
                )
            raise ConcurrencyConflictError(f"ExecutionStep {step_id} update lost race")
        await self._session.flush()
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
