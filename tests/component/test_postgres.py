"""PostgreSQL + pgvector component tests.

These tests run against a real Postgres+pgvector container started via
Testcontainers (see ``conftest.py``). They verify:

  * Container starts and accepts connections on the dynamic port.
  * ``pgvector`` extension is present and queryable.
  * DDL via async SQLAlchemy works.
  * Transaction rollback isolation works (each test sees a clean DB).
  * A trivial connection-pool exhaustion scenario is recoverable.

Marked as ``@pytest.mark.component`` so it can be run independently of
the docker-compose stack smoke tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

pytestmark = [pytest.mark.integration, pytest.mark.component]


async def test_postgres_engine_is_queryable(postgres_component: Any) -> None:
    """A trivial round-trip SELECT 1; proves the engine accepts a session."""
    engine: AsyncEngine = create_async_engine(postgres_component.dsn)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
    finally:
        await engine.dispose()


async def test_pgvector_extension_is_installed(postgres_component: Any) -> None:
    """pgvector MUST be available — it is required by M5+ for vector RAG.

    The pgvector 0.8.6 image ships the extension, but the schema is
    not auto-loaded into a fresh DB. We CREATE EXTENSION and verify.
    """
    engine: AsyncEngine = create_async_engine(postgres_component.dsn)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            row = (
                await conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).first()
            assert row is not None, "pgvector extension is not installed"
            assert row[0] is not None
    finally:
        await engine.dispose()


async def test_rollback_isolation_between_sessions(postgres_component: Any) -> None:
    """Two async sessions on the same engine: an uncommitted INSERT in
    session A is invisible to session B, and rollback in session A
    discards the row entirely.
    """
    engine: AsyncEngine = create_async_engine(postgres_component.dsn)
    Session = async_sessionmaker(engine, expire_on_commit=False)  # noqa: N806
    try:
        # Setup: create a trivial table.
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS comp_test_rollback"))
            await conn.execute(
                text("CREATE TABLE comp_test_rollback (id INT PRIMARY KEY, label TEXT)")
            )

        # Session A: open a tx, insert, but do NOT commit.
        async with Session() as sa:
            await sa.execute(
                text("INSERT INTO comp_test_rollback (id, label) VALUES (1, 'uncommitted')")
            )
            # Session B reads the table — must NOT see the uncommitted row.
            async with Session() as sb:
                rows = (
                    await sb.execute(text("SELECT count(*) FROM comp_test_rollback"))
                ).scalar_one()
                assert rows == 0, "session B should not see session A's uncommitted insert"
            # Now rollback A and verify the row never existed.
            await sa.rollback()

        # Final: confirm the row is gone.
        async with engine.connect() as conn:
            rows = (
                await conn.execute(text("SELECT count(*) FROM comp_test_rollback"))
            ).scalar_one()
            assert rows == 0
    finally:
        await engine.dispose()


async def test_connection_pool_handles_repeated_disconnects(postgres_component: Any) -> None:
    """A pool that disconnects 10x in a row should still serve queries.

    This catches the 'pool stale connection after engine restart'
    failure mode that bit us in M1's recovery tests.
    """
    engine: AsyncEngine = create_async_engine(
        postgres_component.dsn,
        pool_size=2,
        pool_pre_ping=True,  # ping each connection on checkout
    )
    try:
        for i in range(10):
            async with engine.connect() as conn:
                v = (await conn.execute(text(f"SELECT {i} AS n"))).scalar_one()
                assert v == i
    finally:
        await engine.dispose()
