"""PostgreSQL async session factory and unit-of-work.

Provides a shared session factory so infrastructure adapters can create
sessions without importing FastAPI or other app-level context.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Uses lazy initialization to avoid circular import with settings module.
from groundgraph.application.settings import get_settings

_engine = None
_session_factory = None


def _get_engine():
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.postgres_dsn_async,
            pool_size=settings.postgres_pool_size,
            max_overflow=settings.postgres_max_overflow,
            echo=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory  # noqa: PLW0603
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injection friendly session context manager."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def dispose_engine() -> None:
    """Dispose the engine (call on app shutdown)."""
    global _engine, _session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
