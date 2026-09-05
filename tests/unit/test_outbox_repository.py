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
    def __init__(self, rows: list[_Row] | None = None, row: _Row | None = None) -> None:
        self._rows = rows or []
        self._row = row

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

    def add(self, instance: object) -> None:
        self.added.append(cast(_Row, instance))

    async def flush(self) -> None:
        self.flushed += 1

    async def execute(self, statement: Any, parameters: Any = None) -> _Result:
        self.executed.append(statement)
        if self.claim_rows:
            rows = list(self.claim_rows)
            self.claim_rows = []
            return _Result(rows=rows)
        return _Result(rows=[])

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

    session.rows[event.event_id] = _Row(event.event_id)
    loaded = await repo.get(event.event_id)
    assert loaded is not None

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
