"""Unit tests for the Neo4j unit of work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.domain.knowledge import CanonicalEntity
from groundgraph.infrastructure.neo4j.unit_of_work import Neo4jUnitOfWork


@dataclass
class _FakeResult:
    async def single(self) -> dict[str, Any] | None:
        return None


class _FakeTx:
    def __init__(self) -> None:
        self.run_calls: list[tuple[str, dict[str, Any]]] = []
        self.commit_calls = 0
        self.rollback_calls = 0

    async def run(self, query: str, **params: Any) -> _FakeResult:
        self.run_calls.append((query, params))
        return _FakeResult()

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1


class _FakeSession:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.close_calls = 0
        self.tx = _FakeTx()

    async def begin_transaction(self) -> _FakeTx:
        self.begin_calls += 1
        return self.tx

    async def close(self) -> None:
        self.close_calls += 1


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.session_calls = 0
        self.database_args: list[str | None] = []

    def session(self, database: str | None = None) -> _FakeSession:
        self.session_calls += 1
        self.database_args.append(database)
        return self._session


@pytest.mark.asyncio
async def test_uow_commits_graph_writes() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)

    async with Neo4jUnitOfWork(cast(Any, driver), database="neo4j-test") as uow:
        repo = uow.graph
        assert repo is not None
        await repo.create_entity(
            CanonicalEntity(
                entity_id=uuid4(),
                entity_type="Service",
                canonical_name="API",
                aliases=[],
                attributes={},
            )
        )

    assert driver.session_calls == 1
    assert driver.database_args == ["neo4j-test"]
    assert session.begin_calls == 1
    assert session.tx.commit_calls == 1
    assert session.tx.rollback_calls == 0
    assert session.close_calls == 1
    assert len(session.tx.run_calls) == 1


@pytest.mark.asyncio
async def test_uow_rolls_back_graph_writes_on_error() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)

    async def _boom() -> None:
        async with Neo4jUnitOfWork(cast(Any, driver), database="neo4j-test") as uow:
            repo = uow.graph
            assert repo is not None
            await repo.create_entity(
                CanonicalEntity(
                    entity_id=uuid4(),
                    entity_type="Service",
                    canonical_name="API",
                    aliases=[],
                    attributes={},
                )
            )
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _boom()

    assert driver.session_calls == 1
    assert driver.database_args == ["neo4j-test"]
    assert session.begin_calls == 1
    assert session.tx.commit_calls == 0
    assert session.tx.rollback_calls == 1
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_uow_explicit_commit_does_not_finalize_twice() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)

    async with Neo4jUnitOfWork(cast(Any, driver), database="neo4j-test") as uow:
        await uow.commit()

    assert session.tx.commit_calls == 1
    assert session.tx.rollback_calls == 0
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_uow_explicit_rollback_does_not_finalize_twice() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)

    async with Neo4jUnitOfWork(cast(Any, driver), database="neo4j-test") as uow:
        await uow.rollback()

    assert session.tx.commit_calls == 0
    assert session.tx.rollback_calls == 1
    assert session.close_calls == 1
