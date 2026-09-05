"""PostgreSQL outbox consumer with claim/retry semantics (plan.md §2.2 OutboxConsumer port).

Implements the transactional outbox pattern:
- claim_batch: SELECT FOR UPDATE to claim pending events
- mark_completed: mark event as processed
- mark_failed: increment attempts and reset status for retry
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.domain.evidence import OutboxEvent, OutboxEventStatus, OutboxEventType
from groundgraph.infrastructure.postgres.models import Outbox


class OutboxConsumer:
    """PostgreSQL implementation of OutboxConsumer port."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim_batch(self, batch_size: int) -> list[OutboxEvent]:
        result = await self._session.execute(
            select(Outbox)
            .where(Outbox.status == "pending")
            .order_by(Outbox.created_at)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        rows = result.scalars().all()
        if not rows:
            return []

        now = datetime.now(UTC)
        event_ids = [row.event_id for row in rows]

        await self._session.execute(
            update(Outbox)
            .where(Outbox.event_id.in_(event_ids))
            .values(status="processing", claimed_at=now)
        )
        await self._session.commit()

        return [self._sql_to_event(row) for row in rows]

    async def mark_completed(self, event_id: UUID) -> None:
        now = datetime.now(UTC)
        await self._session.execute(
            update(Outbox)
            .where(Outbox.event_id == event_id)
            .values(status="completed", completed_at=now)
        )
        await self._session.commit()

    async def mark_failed(self, event_id: UUID, error: str) -> None:
        result = await self._session.execute(
            select(Outbox).where(Outbox.event_id == event_id).with_for_update()
        )
        row = result.scalar_one_or_none()
        if not row:
            return

        await self._session.execute(
            update(Outbox)
            .where(Outbox.event_id == event_id)
            .values(
                status="pending",
                attempts=row.attempts + 1,
                last_error=error,
                claimed_at=None,
            )
        )
        await self._session.commit()

    def _sql_to_event(self, row: Outbox) -> OutboxEvent:
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
