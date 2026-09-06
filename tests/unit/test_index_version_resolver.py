"""Unit tests for index version resolver."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.infrastructure.postgres.index_version_resolver import PostgresIndexVersionResolver


class _FakeIndexVersionRow:
    def __init__(
        self,
        version_id: Any = None,
        index_name: str = "default",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        is_active: bool = True,
    ) -> None:
        self.version_id = version_id or uuid4()
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.is_active = is_active


class _FakeResult:
    def scalar_one_or_none(self) -> _FakeIndexVersionRow | None:
        return _FakeIndexVersionRow()


class _FakeSession:
    async def execute(self, stmt: Any) -> _FakeResult:
        return _FakeResult()


class _FakeSessionFactory:
    def __call__(self) -> _FakeAsyncContextManager[_FakeSession]:
        return _FakeAsyncContextManager(_FakeSession())


class _FakeAsyncContextManager:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_resolve_active_returns_index_info() -> None:
    factory = cast(Any, _FakeSessionFactory())
    resolver = PostgresIndexVersionResolver(factory)
    result = await resolver.resolve_active("default")
    assert result is not None
    assert result.index_name == "default"
    assert result.embedding_model == "text-embedding-3-small"
    assert result.embedding_dimensions == 1536


@pytest.mark.asyncio
async def test_resolve_active_returns_none_when_not_found() -> None:
    class _EmptyResult:
        def scalar_one_or_none(self) -> None:
            return None

    class _EmptySession:
        async def execute(self, stmt: Any) -> _EmptyResult:
            return _EmptyResult()

    class _EmptyFactory:
        def __call__(self) -> Any:
            return _FakeAsyncContextManager(_EmptySession())

    factory = cast(Any, _EmptyFactory())
    resolver = PostgresIndexVersionResolver(factory)
    result = await resolver.resolve_active("nonexistent")
    assert result is None
