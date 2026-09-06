"""PostgreSQL adapter for IngestionCheckpointRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from groundgraph.application.ports import IngestionCheckpointRepository
from groundgraph.domain.documents import IngestionCheckpoint, IngestionCheckpointStatus
from groundgraph.infrastructure.postgres.models import IngestionCheckpoint as SqlIngestionCheckpoint
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresIngestionCheckpointRepository(IngestionCheckpointRepository):
    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def upsert_checkpoint(  # noqa: PLR0917
        self,
        source_id: UUID,
        canonical_locator: str,
        content_checksum: str,
        status: IngestionCheckpointStatus,
        document_id: UUID | None = None,
        version_id: UUID | None = None,
        error_message: str | None = None,
    ) -> IngestionCheckpoint:
        now = datetime.now(UTC)
        completed_at = (
            now
            if status
            in (
                IngestionCheckpointStatus.PERSISTED,
                IngestionCheckpointStatus.FAILED,
            )
            else None
        )

        values = {
            "source_id": source_id,
            "canonical_locator": canonical_locator,
            "content_checksum": content_checksum,
            "status": status.value,
            "document_id": document_id,
            "version_id": version_id,
            "error_message": error_message,
            "updated_at": now,
        }
        if completed_at is not None:
            values["completed_at"] = completed_at

        stmt = (
            pg_insert(SqlIngestionCheckpoint)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["source_id", "canonical_locator", "content_checksum"],
                set_=values,
            )
            .returning(SqlIngestionCheckpoint)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        return self._to_domain(row)

    async def get_checkpoint(
        self, source_id: UUID, canonical_locator: str, content_checksum: str
    ) -> IngestionCheckpoint | None:
        result = await self._session.execute(
            select(SqlIngestionCheckpoint).where(
                SqlIngestionCheckpoint.source_id == source_id,
                SqlIngestionCheckpoint.canonical_locator == canonical_locator,
                SqlIngestionCheckpoint.content_checksum == content_checksum,
            )
        )
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    def _to_domain(self, row: SqlIngestionCheckpoint) -> IngestionCheckpoint:
        return IngestionCheckpoint(
            checkpoint_id=row.checkpoint_id,
            source_id=row.source_id,
            canonical_locator=row.canonical_locator,
            content_checksum=row.content_checksum,
            status=IngestionCheckpointStatus(row.status),
            document_id=row.document_id,
            version_id=row.version_id,
            error_message=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
        )
