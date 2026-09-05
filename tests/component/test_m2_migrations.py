"""Component test: Alembic migration on a fresh empty database.

Validates that:
- alembic upgrade head succeeds on a truly empty database
- vector extension is created
- all tables, FKs, unique constraints, indexes exist as designed
- downgrade to base works
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from tests.component.conftest import (
    POSTGRES_IMAGE,
    PostgresComponent,
    _wait_postgres_ready,
)

pytestmark = [pytest.mark.integration, pytest.mark.component]


def _run_alembic(dsn: str, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {
        **subprocess.os.environ.copy(),
        "DATABASE_URL": dsn,
    }
    return subprocess.run(
        ["uv", "run", "alembic", *command],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=60,
        check=False,
    )


async def _setup_postgres_container() -> PostgresComponent:
    from testcontainers.community.postgres import PostgresContainer  # noqa: PLC0415

    container = PostgresContainer(
        image=POSTGRES_IMAGE,
        username="groundgraph",
        password="change-me-local-only",
        dbname="groundgraph",
    )
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    comp = PostgresComponent(
        host=host,
        port=int(port),
        user="groundgraph",
        password="change-me-local-only",
        database="groundgraph",
    )
    _wait_postgres_ready(comp, timeout=30.0)
    return comp


async def _assert_table_exists(dsn: str, table: str) -> bool:
    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = :t)"),
            {"t": table},
        )
        exists = result.scalar_one()
    await engine.dispose()
    return exists  # type: ignore[return-value]


async def _assert_index_exists(dsn: str, index: str) -> bool:
    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = :i)"),
            {"i": index},
        )
        exists = result.scalar_one()
    await engine.dispose()
    return exists  # type: ignore[return-value]


async def _assert_vector_extension(dsn: str) -> bool:
    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        vector_ext = result.scalar_one_or_none()
    await engine.dispose()
    return vector_ext == "vector"


async def _count_tables(dsn: str) -> int:
    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'")
        )
        count = result.scalar_one()
    await engine.dispose()
    return count  # type: ignore[return-value]


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.mark.skipif(
    subprocess.run(["sh", "-c", "docker info > /dev/null 2>&1"], check=False).returncode != 0,
    reason="Docker not available",
)
@pytest.mark.asyncio
async def test_alembic_upgrade_head_on_empty_db(project_root: Path) -> None:
    comp = await _setup_postgres_container()
    dsn = comp.dsn

    upgrade_result = _run_alembic(dsn, ["upgrade", "head"], cwd=project_root)
    assert upgrade_result.returncode == 0, (
        f"alembic upgrade head failed:\n"
        f"stdout: {upgrade_result.stdout}\n"
        f"stderr: {upgrade_result.stderr}"
    )

    assert await _assert_table_exists(dsn, "sources"), "sources table not created"
    assert await _assert_table_exists(dsn, "documents"), "documents table not created"
    assert await _assert_table_exists(dsn, "document_versions"), (
        "document_versions table not created"
    )
    assert await _assert_table_exists(dsn, "chunks"), "chunks table not created"
    assert await _assert_table_exists(dsn, "chunk_embeddings"), "chunk_embeddings table not created"
    assert await _assert_table_exists(dsn, "execution_runs"), "execution_runs table not created"
    assert await _assert_table_exists(dsn, "execution_steps"), "execution_steps table not created"
    assert await _assert_table_exists(dsn, "outbox"), "outbox table not created"
    assert await _assert_table_exists(dsn, "index_versions"), "index_versions table not created"

    assert await _assert_vector_extension(dsn), "vector extension not created"

    assert await _assert_index_exists(dsn, "ix_execution_runs_tenant_id"), "tenant_id index missing"
    assert await _assert_index_exists(dsn, "ix_outbox_status_available_created"), (
        "outbox status index missing"
    )

    downgrade_result = _run_alembic(dsn, ["downgrade", "base"], cwd=project_root)
    assert downgrade_result.returncode == 0, (
        f"alembic downgrade base failed:\n"
        f"stdout: {downgrade_result.stdout}\n"
        f"stderr: {downgrade_result.stderr}"
    )

    table_count = await _count_tables(dsn)
    assert table_count == 1, (
        f"Expected only alembic_version after downgrade, found {table_count} tables"
    )


@pytest.mark.skipif(
    subprocess.run(["sh", "-c", "docker info > /dev/null 2>&1"], check=False).returncode != 0,
    reason="Docker not available",
)
@pytest.mark.asyncio
async def test_migration_vector_extension(project_root: Path) -> None:
    comp = await _setup_postgres_container()
    dsn = comp.dsn

    upgrade_result = _run_alembic(dsn, ["upgrade", "head"], cwd=project_root)
    assert upgrade_result.returncode == 0, (
        f"alembic upgrade head failed:\n"
        f"stdout: {upgrade_result.stdout}\n"
        f"stderr: {upgrade_result.stderr}"
    )

    engine = create_async_engine(dsn)
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        version = result.scalar_one_or_none()
        assert version is not None, "vector extension not found"
    await engine.dispose()
