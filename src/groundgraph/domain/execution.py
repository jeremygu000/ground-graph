"""Execution domain contracts (plan.md §4 + §0.6 execution_run/step).

Pure Pydantic v2 — no infrastructure imports.

The execution model is the third graph the system maintains: the
vertical knowledge graph (Neo4j), the horizontal workflow graph
(LangGraph), and the queryable execution graph (PostgreSQL). M2
defines the contract; the workflows adapter in M7+ writes here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from groundgraph.domain.defaults import empty_json_dict, empty_uuid_list
from groundgraph.domain.types import validate_json_value


class ExecutionRunStatus(StrEnum):
    """The terminal-allowed state machine for an execution run.

    Allowed transitions (anything not in this table is rejected
    by the SQLAlchemy+Neo4j adapters in M7+):

        PENDING  -> RUNNING
        RUNNING  -> SUCCEEDED | FAILED | CANCELLED
        PENDING  -> CANCELLED
        SUCCEEDED | FAILED | CANCELLED are terminal.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Static transition table — used by the adapter in
# ``src/groundgraph/infrastructure/postgres/execution_store.py`` to
# reject invalid transitions in SQL.
ALLOWED_RUN_TRANSITIONS: dict[ExecutionRunStatus, set[ExecutionRunStatus]] = {
    ExecutionRunStatus.PENDING: {ExecutionRunStatus.RUNNING, ExecutionRunStatus.CANCELLED},
    ExecutionRunStatus.RUNNING: {
        ExecutionRunStatus.SUCCEEDED,
        ExecutionRunStatus.FAILED,
        ExecutionRunStatus.CANCELLED,
    },
    ExecutionRunStatus.SUCCEEDED: set(),
    ExecutionRunStatus.FAILED: set(),
    ExecutionRunStatus.CANCELLED: set(),
}


class ExecutionStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


ALLOWED_STEP_TRANSITIONS: dict[ExecutionStepStatus, set[ExecutionStepStatus]] = {
    ExecutionStepStatus.PENDING: {ExecutionStepStatus.RUNNING, ExecutionStepStatus.SKIPPED},
    ExecutionStepStatus.RUNNING: {
        ExecutionStepStatus.SUCCEEDED,
        ExecutionStepStatus.FAILED,
        ExecutionStepStatus.SKIPPED,
    },
    ExecutionStepStatus.SUCCEEDED: set(),
    ExecutionStepStatus.FAILED: set(),
    ExecutionStepStatus.SKIPPED: set(),
}


class ExecutionRun(BaseModel):
    """One query / one workflow invocation, top to bottom.

    The run is the row you join every other table to when
    reproducing a result. ``tenant_id`` scopes the run; combined
    with the principal list it is the cross-cutting ACL.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    run_id: UUID
    workflow: str  # e.g. "query", "ingest", "evaluate"
    status: ExecutionRunStatus
    principal: str
    tenant_id: str
    input: dict[str, object] = Field(default_factory=empty_json_dict)
    output: dict[str, object] = Field(default_factory=empty_json_dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_terminal_consistency(self) -> ExecutionRun:
        terminal = (
            ExecutionRunStatus.SUCCEEDED,
            ExecutionRunStatus.FAILED,
            ExecutionRunStatus.CANCELLED,
        )
        if self.status in terminal and self.finished_at is None:
            raise ValueError(
                f"execution_run in terminal status {self.status!r} must have finished_at set"
            )
        return self

    @field_validator("input", "output", mode="before")
    @classmethod
    def _validate_json_fields(cls, value: object) -> object:
        return validate_json_value(value)


class ExecutionStep(BaseModel):
    """One node in the workflow graph.

    Steps form a DAG via ``depends_on``. The plan of record is
    the workflow definition (LangGraph) but the audit record lives
    here so the user can ask "what actually happened" after the
    fact.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    step_id: UUID
    run_id: UUID
    name: str
    status: ExecutionStepStatus
    attempt: int = Field(default=1, ge=1)
    depends_on: list[UUID] = Field(default_factory=empty_uuid_list)
    input: dict[str, object] = Field(default_factory=empty_json_dict)
    output: dict[str, object] = Field(default_factory=empty_json_dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None

    @field_validator("input", "output", mode="before")
    @classmethod
    def _validate_json_fields_step(cls, value: object) -> object:
        return validate_json_value(value)


def assert_run_transition(current: ExecutionRunStatus, target: ExecutionRunStatus) -> None:
    """Raise ``ValueError`` if ``current -> target`` is not a legal transition.

    Imported by infrastructure adapters so the same state machine
    is enforced on every write.
    """
    if target not in ALLOWED_RUN_TRANSITIONS[current]:
        raise ValueError(f"illegal execution_run transition: {current.value} -> {target.value}")
