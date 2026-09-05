"""Component-level integration test fixtures.

Component tests target a SINGLE dependency in isolation, brought up via
Testcontainers rather than the full docker-compose stack. This is much
cheaper to run, hermetic (no shared state with other tests), and lets
us pin a specific image version per fixture.

Layering (see ADR-010-testcontainers-component-foundation.md):

  ┌─────────────────────────────────────────────────────────────┐
  │ Stack smoke tests        → tests/integration/test_m1_stack.py│
  │ (full docker-compose, retained as M1 readiness evidence)   │
  └─────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────┐
  │ Component tests           → tests/component/                 │
  │ (Testcontainers, single dependency, dynamic ports)          │
  └─────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────┐
  │ Unit tests                → tests/unit/                      │
  │ (no Docker, fake adapters)                                  │
  └─────────────────────────────────────────────────────────────┘

Skip policy: if Docker is not available (CLI missing or daemon
unreachable), every component test is skipped, never failed. This
matches the existing stack fixture's policy.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from groundgraph.infrastructure.postgres.models import Base

# Pinned image versions — MUST match docker-compose.yml so component
# tests exercise the same engine as the stack smoke tests.
POSTGRES_IMAGE = "pgvector/pgvector:0.8.6-pg16"
NEO4J_IMAGE = "neo4j:5.26.10-community"


@dataclass
class PostgresComponent:
    """Connection parameters for a Testcontainers Postgres + pgvector."""

    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN (postgresql+asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def sync_dsn(self) -> str:
        """Sync psycopg2-style DSN, used by Alembic."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class Neo4jComponent:
    """Connection parameters for a Testcontainers Neo4j."""

    uri: str
    user: str
    password: str

    @property
    def http_uri(self) -> str:
        return self.uri.replace("bolt://", "http://").replace(":7687", ":7474")


def _docker_available() -> tuple[bool, str]:
    """Return (ok, reason). ok=True iff docker CLI exists AND daemon is reachable."""
    if shutil.which("docker") is None:
        return False, "Docker CLI is not available on PATH."
    info = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if info.returncode != 0:
        return False, f"Docker daemon is not reachable: {info.stderr or info.stdout}"
    return True, ""


# ---------------------------------------------------------------------------
# Session-scoped component fixtures
#
# Scope: `session` so a single container is reused across all component
# tests in the run. The containers are cheap to keep alive (~256 MB RSS
# for Postgres, ~512 MB for Neo4j) compared to the 60-90s cost of
# starting a fresh container per test function.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Skip signal: True iff Docker is reachable on this host."""
    ok, _ = _docker_available()
    return ok


@pytest.fixture(scope="session")
def postgres_component(docker_available: bool) -> Generator[PostgresComponent, None, None]:
    """Start a single Postgres+pgvector container for the whole test session.

    The container uses Testcontainers' dynamic port allocation, so
    multiple CI jobs on the same host won't collide. Image is pinned
    to match docker-compose.yml so we exercise the same engine
    version as the stack smoke tests.
    """
    if not docker_available:
        pytest.skip("Docker is not available; skipping component test.")

    from testcontainers.community.postgres import PostgresContainer  # noqa: PLC0415

    # pgvector 0.8.6 ships on top of PG 16.
    container = PostgresContainer(
        image=POSTGRES_IMAGE,
        username="groundgraph",
        password="change-me-local-only",
        dbname="groundgraph",
    )
    container.start()

    try:
        # PostgresContainer.get_connection_url() returns a JDBC-style
        # URL; we extract host/port via the get_exposed_port helper.
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        comp = PostgresComponent(
            host=host,
            port=int(port),
            user="groundgraph",
            password="change-me-local-only",
            database="groundgraph",
        )
        # Wait for the engine to be queryable.
        _wait_postgres_ready(comp, timeout=30.0)
        yield comp
    finally:
        container.stop()


@pytest.fixture(scope="session")
async def postgres(
    postgres_component: PostgresComponent,
) -> AsyncGenerator[dict[str, object], None]:
    """PostgresComponent with schema migrated.

    Creates all tables via SQLAlchemy Base.metadata.create_all once per session,
    then yields a dict with:
    - ``dsn``: the async SQLAlchemy connection string
    - ``session``: an async session factory (async_sessionmaker)
    This lets tests use the full schema without managing migrations.
    """
    async_engine = create_async_engine(postgres_component.dsn)

    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    try:
        yield {
            "dsn": postgres_component.dsn,
            "session": async_session_factory,
        }
    finally:
        await async_engine.dispose()


@pytest.fixture(scope="session")
def neo4j_component(docker_available: bool) -> Generator[Neo4jComponent, None, None]:
    """Start a single Neo4j container for the whole test session.

    Neo4j's container has its own startup dance — it runs a `warmed-up`
    dance on first boot, plus a per-neo4j-version bolt-port. The
    Testcontainers Neo4jContainer handles auth and ports.
    """
    if not docker_available:
        pytest.skip("Docker is not available; skipping component test.")

    from testcontainers.community.neo4j import Neo4jContainer  # noqa: PLC0415

    container = Neo4jContainer(image=NEO4J_IMAGE, password="change-me-local-only")
    container.start()

    try:
        host = container.get_container_host_ip()
        bolt_port = container.get_exposed_port(7687)
        comp = Neo4jComponent(
            uri=f"bolt://{host}:{int(bolt_port)}",
            user="neo4j",
            password="change-me-local-only",
        )
        _wait_neo4j_ready(comp, timeout=60.0)
        yield comp
    finally:
        container.stop()


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def _wait_postgres_ready(comp: PostgresComponent, timeout: float) -> None:
    """Block until Postgres accepts connections or timeout expires."""
    import socket  # noqa: PLC0415

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((comp.host, comp.port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Postgres not reachable at {comp.host}:{comp.port} within {timeout}s")


def _wait_neo4j_ready(comp: Neo4jComponent, timeout: float) -> None:
    """Block until Neo4j accepts bolt connections or timeout expires.

    We open a real neo4j driver connection to verify the bolt protocol
    is up — TCP-open is not enough because Neo4j's HTTP/bolt listeners
    come up in stages.
    """
    try:
        from neo4j import GraphDatabase  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("neo4j driver not installed; component test cannot run") from exc

    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            driver = GraphDatabase.driver(comp.uri, auth=(comp.user, comp.password))
        except Exception as exc:
            last_err = exc
            time.sleep(1.0)
            continue
        else:
            try:
                driver.verify_connectivity()
            finally:
                driver.close()
            return
    raise RuntimeError(f"Neo4j not reachable at {comp.uri} within {timeout}s: {last_err}")
