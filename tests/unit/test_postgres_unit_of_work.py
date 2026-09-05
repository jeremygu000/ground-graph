"""Unit tests for the PostgreSQL unit of work."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.infrastructure.postgres.unit_of_work import PostgresUnitOfWork


class _FakeSession:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def add(self, instance: object) -> None:
        return None

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def get(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_uow_commits_on_success() -> None:
    session = _FakeSession()
    session_factory = cast(Callable[[], AsyncSession], lambda: session)
    uow = PostgresUnitOfWork(session_factory)

    async with uow:
        assert uow.execution is not None

    assert session.commit_calls == 1
    assert session.rollback_calls == 0
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_uow_rolls_back_on_error() -> None:
    session = _FakeSession()
    session_factory = cast(Callable[[], AsyncSession], lambda: session)
    uow = PostgresUnitOfWork(session_factory)

    with pytest.raises(RuntimeError):
        async with uow:
            raise RuntimeError("boom")

    assert session.commit_calls == 0
    assert session.rollback_calls == 1
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_uow_manual_commit_and_rollback() -> None:
    session = _FakeSession()
    session_factory = cast(Callable[[], AsyncSession], lambda: session)
    uow = PostgresUnitOfWork(session_factory)

    async with uow:
        await uow.commit()
        await uow.rollback()

    assert session.commit_calls >= 1
    assert session.rollback_calls >= 1
