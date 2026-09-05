"""PostgreSQL transactional outbox repository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select

from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.infrastructure.postgres.models import Outbox as OutboxModel
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresOutboxRepository:
    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def add(self, event: OutboxEvent) -> OutboxEvent:
        model = OutboxModel(
            event_id=event.event_id,
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            event_type=event.event_type.value,
            payload=event.payload,
            status=event.status.value,
            attempts=event.attempts,
            last_error=event.last_error,
            created_at=event.created_at,
            available_at=event.created_at,
            claimed_at=event.claimed_at,
            completed_at=event.completed_at,
        )
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get(self, event_id: UUID) -> OutboxEvent | None:
        row = await self._session.get(OutboxModel, event_id)
        return self._to_domain(row) if row else None

    async def claim_batch(
        self, batch_size: int, worker_id: str, lease_duration_seconds: int
    ) -> list[OutboxEvent]:
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(OutboxModel)
            .where(OutboxModel.status == OutboxEventStatus.PENDING.value)
            .where(OutboxModel.available_at <= now)
            .order_by(OutboxModel.created_at)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        rows = result.scalars().all()
        if not rows:
            return []

        lease_expires_at = now + timedelta(seconds=lease_duration_seconds)
        claimed: list[OutboxEvent] = []
        for row in rows:
            claim_token = uuid4().hex
            row.status = OutboxEventStatus.CLAIMED.value
            row.claimed_by = worker_id
            row.claim_token = claim_token
            row.claimed_at = now
            row.lease_expires_at = lease_expires_at
            row.attempts += 1
            claimed.append(self._to_domain(row))
        await self._session.flush()
        return claimed

    async def mark_completed(self, event_id: UUID, claim_token: str) -> None:
        row = await self._session.get(OutboxModel, event_id)
        if not row or row.claim_token != claim_token:
            raise ValueError("invalid claim token")
        row.status = OutboxEventStatus.COMPLETED.value
        row.completed_at = datetime.now(UTC)
        row.lease_expires_at = None
        await self._session.flush()

    async def mark_failed(self, event_id: UUID, claim_token: str, error: str) -> None:
        row = await self._session.get(OutboxModel, event_id)
        if not row or row.claim_token != claim_token:
            raise ValueError("invalid claim token")
        row.status = OutboxEventStatus.PENDING.value
        row.last_error = error[:512]
        row.claimed_by = None
        row.claim_token = None
        row.claimed_at = None
        row.lease_expires_at = None
        row.available_at = datetime.now(UTC) + timedelta(seconds=min(60, 2**row.attempts))
        await self._session.flush()

    def _to_domain(self, row: OutboxModel) -> OutboxEvent:
        return OutboxEvent(
            event_id=row.event_id,
            aggregate_type=row.aggregate_type,
            aggregate_id=row.aggregate_id,
            event_type=OutboxEventType(row.event_type),
            payload=row.payload,
            status=OutboxEventStatus(row.status),
            attempts=row.attempts,
            last_error=row.last_error,
            created_at=row.created_at,
            claimed_at=row.claimed_at,
            completed_at=row.completed_at,
        )
