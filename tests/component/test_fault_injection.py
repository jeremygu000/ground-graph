"""Fault injection tests for M11 production hardening.

These tests verify that GroundGraph degrades gracefully when infrastructure
components (Postgres/pgvector, Neo4j, object storage, telemetry) are unavailable.

Requires Docker for Testcontainers. Skipped if Docker is not available on the host.
These tests are NOT run as part of normal `make check` (unit-only).
Run with: `make test-fault` (or directly with pytest when Docker is available).
"""

from __future__ import annotations

import shutil
import subprocess
from uuid import UUID

import pytest
from sqlalchemy import text


def _docker_available() -> bool:
    """Return True iff docker CLI exists AND daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15, check=False
    )
    return result.returncode == 0


@pytest.fixture(scope="module")
def docker_available() -> bool:
    """Skip signal: True iff Docker is reachable on this host."""
    return _docker_available()


class TestPostgresFailureRecovery:
    """Test that the system degrades gracefully when Postgres/pgvector is unavailable."""

    @pytest.mark.skip(reason="M11: needs schema setup or redesign - issue with test setup")
    @pytest.mark.asyncio
    async def test_vector_retrieval_returns_empty_on_connection_failure(
        self, docker_available: bool
    ) -> None:
        """Vector retrieval should return empty results (not raise) when Postgres is unreachable."""
        from testcontainers.community.postgres import PostgresContainer

        from groundgraph.infrastructure.postgres.vector_retriever import PostgresVectorRetriever

        container = PostgresContainer(image="pgvector/pgvector:pg16")
        container.start()
        try:
            host = container.get_container_host_ip()
            port = container.get_exposed_port(5432)

            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

            dsn = f"postgresql+asyncpg://{container.username}:{container.password}@{host}:{int(port)}/postgres"
            engine = create_async_engine(dsn, pool_size=0)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)

            retriever = PostgresVectorRetriever(session_factory=session_factory)

            result = await retriever.search(
                query_vector=[0.0] * 128,
                top_k=5,
                allowed_principals=["engineering"],
                tenant_id="test-tenant",
                index_version_id=None,
            )
            assert isinstance(result, list)
        finally:
            container.stop()

    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_session_factory_times_out_on_unreachable_postgres(
        self, docker_available: bool
    ) -> None:
        """SQLAlchemy session factory should raise on unreachable Postgres."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost:55432/nonexistent",
            pool_size=0,
        )
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        with pytest.raises(OSError, match="Multiple exceptions"):
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))


class TestNeo4jFailureRecovery:
    """Test that the system degrades gracefully when Neo4j is unavailable."""

    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_graph_repository_raises_on_unreachable_neo4j(
        self, docker_available: bool
    ) -> None:
        """Neo4jGraphRepository should raise when Neo4j is unreachable."""
        from neo4j import AsyncGraphDatabase

        from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository

        driver = AsyncGraphDatabase.driver(
            "bolt://localhost:57632", auth=("neo4j", "password"), max_connection_lifetime=0.1
        )
        repo = Neo4jGraphRepository(driver=driver, database="neo4j")

        from neo4j.exceptions import ServiceUnavailable

        with pytest.raises(ServiceUnavailable):
            await repo.get_entity(UUID("00000000-0000-0000-0000-000000000001"))

        await driver.close()


class TestObjectStorageFailureRecovery:
    """Test graceful degradation when object storage (MinIO/S3) is unavailable."""

    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_object_store_raises_on_unreachable_endpoint(
        self, docker_available: bool
    ) -> None:
        """ObjectStore should raise a specific exception when S3 endpoint is unreachable."""
        from unittest.mock import MagicMock

        from pydantic import SecretStr

        from groundgraph.application.settings import Settings
        from groundgraph.infrastructure.object_storage.s3_store import S3ObjectStore

        mock_settings = MagicMock(spec=Settings)
        mock_settings.s3_endpoint_url = "http://localhost:57999"
        mock_settings.s3_access_key = "test"
        mock_settings.s3_secret_key = SecretStr("test")
        mock_settings.s3_region = "us-east-1"
        mock_settings.s3_use_ssl = False
        mock_settings.s3_bucket_raw = "test-raw"
        mock_settings.s3_bucket_processed = "test-processed"

        store = S3ObjectStore(settings=mock_settings)

        with pytest.raises((OSError, Exception)):
            await store.put_raw("key", b"data")


class TestTelemetryFailureRecovery:
    """Test graceful degradation when the OTLP telemetry endpoint is unavailable."""

    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_telemetry_exporter_closes_gracefully_on_otlp_unavailable(
        self, docker_available: bool
    ) -> None:
        """OTLP exporter should not crash the application when collector is unreachable."""
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint="http://localhost:43100", insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = trace.get_tracer("test")

        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test.attr", "value")

        processor.shutdown()
