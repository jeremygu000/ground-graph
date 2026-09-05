"""Unit tests for the PostgreSQL outbox repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.infrastructure.postgres.outbox_repository import PostgresOutboxRepository
from groundgraph.infrastructure.postgres.session import PostgresSession


@dataclass
class _Row:
    event_id: UUID
    aggregate_type: str = "document"
    aggregate_id: UUID = field(default_factory=uuid4)
    event_type: str = OutboxEventType.DOCUMENT_PARSED.value
    payload: dict[str, Any] = field(default_factory=lambda: {"title": "x"})
    status: str = OutboxEventStatus.PENDING.value
    attempts: int = 0
    last_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    claimed_at: datetime | None = None
    completed_at: datetime | None = None
    claimed_by: str | None = None
    claim_token: str | None = None
    lease_expires_at: datetime | None = None


class _Result:
    def __init__(
        self, rows: list[_Row] | None = None, row: _Row | None = None, rowcount: int = 0
    ) -> None:
        self._rows = rows or []
        self._row = row
        self.rowcount = rowcount

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[_Row]:
        return self._rows

    def scalar_one_or_none(self) -> _Row | None:
        return self._row


class _Session:
    def __init__(self) -> None:
        self.added: list[_Row] = []
        self.flushed = 0
        self.executed: list[Any] = []
        self.rows: dict[UUID, _Row] = {}
        self.claim_rows: list[_Row] = []
        self._last_claim_token: str | None = None

    def add(self, instance: object) -> None:
        self.added.append(cast(_Row, instance))

    async def flush(self) -> None:
        self.flushed += 1

    async def execute(self, statement: Any, parameters: Any | None = None) -> _Result:
        self.executed.append(statement)
        if parameters:
            self.executed_params = parameters
        if self.claim_rows:
            return self._claim_rows()
        if hasattr(statement, "_values") and hasattr(statement, "_where_criteria"):
            updated = self._apply_update(statement)
            return _Result(rowcount=1 if updated else 0)
        return _Result(rowcount=1)

    def _claim_rows(self) -> _Result:
        rows = list(self.claim_rows)
        self.claim_rows = []
        return _Result(rows=rows, rowcount=len(rows))

    def _apply_update(self, statement: Any) -> bool:
        target_event_id = self._target_event_id(statement)
        if not target_event_id:
            return False
        row = self.rows.get(cast(UUID, target_event_id))
        if not row:
            return False
        if self._fencing_failed(statement, row):
            return False
        for col, raw_value in getattr(statement, "_values", {}).items():
            col_name = self._column_name(col)
            if not col_name:
                continue
            value = raw_value.value if hasattr(raw_value, "value") else raw_value
            if hasattr(row, col_name):
                setattr(row, col_name, value)
        return True

    def _fencing_failed(self, statement: Any, row: _Row) -> bool:
        criteria = getattr(statement, "_where_criteria", ())
        wanted_status = OutboxEventStatus.CLAIMED.value
        status_ok = any(
            getattr(getattr(clause, "left", None), "key", None) == "status"
            and getattr(getattr(clause, "right", None), "value", getattr(clause, "right", None))
            == wanted_status
            for clause in criteria
        )
        token_ok = any(
            getattr(getattr(clause, "left", None), "key", None) == "claim_token"
            and getattr(getattr(clause, "right", None), "value", getattr(clause, "right", None))
            == row.claim_token
            for clause in criteria
        )
        return not (status_ok and token_ok)

    def _target_event_id(self, statement: Any) -> UUID | None:
        for clause in statement._where_criteria:
            if not hasattr(clause, "left") or not hasattr(clause, "right"):
                continue
            left = clause.left
            if hasattr(left, "key") and left.key == "event_id":
                right = clause.right
                value = right.value if hasattr(right, "value") else right
                return cast(UUID, value)
        return None

    def _column_name(self, column: Any) -> str | None:
        col_str = column if isinstance(column, str) else getattr(column, "key", None)
        if not col_str:
            return None
        return col_str.split(".")[-1]

    async def get(self, entity: Any, ident: Any) -> _Row | None:
        return self.rows.get(cast(UUID, ident))


def _execute_row(row: _Row) -> Any:
    async def _execute(_statement: Any, _parameters: Any = None) -> _Result:
        return _Result(row=row)

    return _execute


@pytest.mark.asyncio
async def test_add_get_and_empty_claim() -> None:
    session = _Session()
    repo = PostgresOutboxRepository(cast(PostgresSession, session))
    completed_at = datetime.now(UTC)
    event = OutboxEvent(
        event_id=uuid4(),
        aggregate_type="document",
        aggregate_id=uuid4(),
        event_type=OutboxEventType.DOCUMENT_PARSED,
        payload={"title": "x"},
        created_at=datetime.now(UTC),
        completed_at=completed_at,
    )

    saved = await repo.add(event)
    assert saved.event_id == event.event_id
    assert saved.completed_at == completed_at

    session.rows[event.event_id] = _Row(event.event_id)
    session.rows[event.event_id].completed_at = completed_at
    loaded = await repo.get(event.event_id)
    assert loaded is not None
    assert loaded.completed_at == completed_at

    assert await repo.claim_batch(1, "worker", 30) == []


@pytest.mark.asyncio
async def test_claim_and_fail_paths() -> None:
    session = _Session()
    row = _Row(event_id=uuid4())
    session.rows[row.event_id] = row
    repo = PostgresOutboxRepository(cast(PostgresSession, session))

    session.claim_rows = [row]
    claimed = await repo.claim_batch(1, "worker-a", 30)
    assert len(claimed) == 1
    token = row.claim_token or ""

    with pytest.raises(ValueError, match="invalid claim token"):
        await repo.mark_completed(row.event_id, "wrong")

    with pytest.raises(ValueError, match="invalid claim token"):
        await repo.mark_failed(row.event_id, "wrong", "nope")

    await repo.mark_failed(row.event_id, token, "x" * 600)
    assert len(row.last_error or "") == 512


@pytest.mark.asyncio
async def test_outbox_repository_rejects_mutated_payload() -> None:
    session = _Session()
    repo = PostgresOutboxRepository(cast(PostgresSession, session))
    payload: dict[str, object] = {"ok": {"nested": "value"}}
    event = OutboxEvent(
        event_id=uuid4(),
        aggregate_type="document",
        aggregate_id=uuid4(),
        event_type=OutboxEventType.DOCUMENT_PARSED,
        payload=payload,
        created_at=datetime.now(UTC),
    )
    cast(dict[str, object], event.payload["ok"])["blob"] = b"raw"

    with pytest.raises(ValueError, match="JsonValue"):
        await repo.add(event)

    assert session.added == []
    assert session.flushed == 0
