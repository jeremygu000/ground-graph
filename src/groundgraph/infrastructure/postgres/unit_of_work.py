"""PostgreSQL transactional unit of work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from groundgraph.infrastructure.postgres.document_repository import (
    PostgresDocumentRepository,
)
from groundgraph.infrastructure.postgres.execution_store import ExecutionRepository
from groundgraph.infrastructure.postgres.outbox_repository import PostgresOutboxRepository
from groundgraph.infrastructure.postgres.session import PostgresSession


class PostgresUnitOfWork:
    """Async SQLAlchemy-backed unit of work.

    It owns the transaction boundary; repositories must not commit.
    """

    def __init__(self, session_factory: Callable[[], PostgresSession]) -> None:
        self._session_factory = session_factory
        self._session: PostgresSession | None = None
        self.documents: PostgresDocumentRepository | None = None
        self.outbox: PostgresOutboxRepository | None = None
        self.execution: ExecutionRepository | None = None

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        self.documents = PostgresDocumentRepository(self._session)
        self.outbox = PostgresOutboxRepository(self._session)
        self.execution = ExecutionRepository(self._session)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work not started")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work not started")
        await self._session.rollback()
