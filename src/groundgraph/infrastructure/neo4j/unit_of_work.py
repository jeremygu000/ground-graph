"""Neo4j transactional unit of work."""

from __future__ import annotations

from typing import Any, Self, cast

from neo4j import AsyncDriver

from groundgraph.application.settings import get_settings
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository


class Neo4jUnitOfWork:
    """Async Neo4j-backed unit of work.

    It owns the session and transaction boundary; repositories created
    through this UoW are bound to the same transaction.
    """

    def __init__(self, driver: AsyncDriver, database: str | None = None) -> None:
        self._driver = driver
        self._database = database or get_settings().neo4j_database
        self._session: Any | None = None
        self._tx: Any | None = None
        self.graph: Neo4jGraphRepository | None = None

    async def __aenter__(self) -> Self:
        session = cast(Any, self._driver.session(database=self._database))
        self._session = session
        self._tx = await session.begin_transaction()
        self.graph = Neo4jGraphRepository(self._driver, tx=self._tx)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is None or self._tx is None:
            return
        try:
            if exc_type is None:
                await self._tx.commit()
            else:
                await self._tx.rollback()
        finally:
            await self._session.close()
            self._session = None
            self._tx = None
            self.graph = None

    async def commit(self) -> None:
        if self._tx is None:
            raise RuntimeError("unit of work not started")
        await self._tx.commit()

    async def rollback(self) -> None:
        if self._tx is None:
            raise RuntimeError("unit of work not started")
        await self._tx.rollback()
