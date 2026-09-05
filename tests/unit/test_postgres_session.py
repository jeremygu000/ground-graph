"""Unit tests for the PostgreSQL session helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.infrastructure.postgres import session as postgres_session


class _FakeEngine:
    def __init__(self) -> None:
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


class _FakeAsyncSession:
    def __init__(self) -> None:
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
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1


class _SessionContext:
    def __init__(self, session: _FakeAsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeAsyncSession:
        return self._session

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _SessionView:
    def __init__(self, session: object) -> None:
        self._session = cast(_FakeAsyncSession, session)

    @property
    def close_calls(self) -> int:
        return self._session.close_calls


class _Factory:
    def __init__(self, session: _FakeAsyncSession) -> None:
        self._session = session
        self.calls = 0

    def __call__(self) -> _SessionContext:
        self.calls += 1
        return _SessionContext(self._session)


def test_session_factory_type_aliases() -> None:
    assert postgres_session.PostgresSession is not None


def test_get_engine_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        postgres_session,
        "get_settings",
        lambda: SimpleNamespace(
            postgres_dsn_async="postgresql+asyncpg://user:pass@localhost/db",
            postgres_pool_size=1,
            postgres_max_overflow=0,
        ),
    )
    monkeypatch.setattr(postgres_session, "_engine", None)

    engine = postgres_session._get_engine()
    assert engine is not None


def test_postgres_session_protocol_is_structural() -> None:
    assert hasattr(AsyncSession, "execute")


@pytest.mark.asyncio
async def test_dispose_engine_clears_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = _FakeEngine()
    fake_factory = object()
    monkeypatch.setattr(postgres_session, "_engine", fake_engine)
    monkeypatch.setattr(postgres_session, "_session_factory", fake_factory)

    await postgres_session.dispose_engine()

    assert fake_engine.dispose_calls == 1
    assert postgres_session._engine is None
    assert postgres_session._session_factory is None


def test_get_session_factory_reuses_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postgres_session, "_engine", object())
    monkeypatch.setattr(postgres_session, "_session_factory", None)

    factory = postgres_session.get_session_factory()
    assert factory is postgres_session.get_session_factory()


@pytest.mark.asyncio
async def test_get_session_closes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeAsyncSession()
    factory = _Factory(session)
    monkeypatch.setattr(postgres_session, "get_session_factory", lambda: factory)

    async with postgres_session.get_session() as yielded:
        view = _SessionView(yielded)
        assert view.close_calls == 0

    assert factory.calls == 1
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_dispose_engine_noop_when_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postgres_session, "_engine", None)
    monkeypatch.setattr(postgres_session, "_session_factory", None)

    await postgres_session.dispose_engine()

    assert postgres_session._engine is None
    assert postgres_session._session_factory is None
