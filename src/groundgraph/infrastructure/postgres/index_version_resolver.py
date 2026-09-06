"""PostgreSQL implementation of the IndexVersionResolver port."""

from __future__ import annotations

from sqlalchemy import select

from groundgraph.application.ports import IndexVersionInfo, IndexVersionResolver
from groundgraph.infrastructure.postgres.models import IndexVersion as IndexVersionModel
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresIndexVersionResolver(IndexVersionResolver):
    """Resolve active IndexVersion from PostgreSQL."""

    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def resolve_active(self, index_name: str) -> IndexVersionInfo | None:
        stmt = (
            select(IndexVersionModel)
            .where(
                IndexVersionModel.index_name == index_name,
                IndexVersionModel.is_active == True,  # noqa: E712
            )
            .order_by(IndexVersionModel.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return IndexVersionInfo(
            version_id=row.version_id,
            index_name=row.index_name,
            embedding_model=row.embedding_model,
            embedding_dimensions=row.embedding_dimensions,
            is_active=row.is_active,
        )
