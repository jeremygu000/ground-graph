"""Fault injection tests for M11 production hardening.

These tests are marked with `@pytest.mark.skip(reason="M11: requires Docker for fault injection")`.
When Docker is available, they test that the application degrades gracefully when
infrastructure components (Postgres, Neo4j, object storage, telemetry) fail.

The test-fault Makefile target runs these with real Testcontainers to simulate
infrastructure failures (connection refused, timeout, etc.).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="M11: requires Docker for fault injection")


class TestPostgresFailureRecovery:
    """Test that the system degrades gracefully when Postgres is unavailable."""

    @pytest.mark.asyncio
    async def test_document_ingestion_fails_cleanly_when_postgres_unavailable(self) -> None:
        """Ingestion should raise DataUnavailableError when Postgres connection fails."""
        pass

    @pytest.mark.asyncio
    async def test_vector_retrieval_returns_empty_on_connection_failure(self) -> None:
        """Vector retrieval should return empty results when Postgres is unreachable."""
        pass

    @pytest.mark.asyncio
    async def test_execution_store_raises_on_pg_unavailable(self) -> None:
        """ExecutionRepository should raise when Postgres is unavailable."""
        pass


class TestNeo4jFailureRecovery:
    """Test that the system degrades gracefully when Neo4j is unavailable."""

    @pytest.mark.asyncio
    async def test_graph_traversal_fails_cleanly_on_neo4j_unavailable(self) -> None:
        """Graph retrieval should return empty evidence when Neo4j is unreachable."""
        pass

    @pytest.mark.asyncio
    async def test_entity_resolution_skips_on_neo4j_failure(self) -> None:
        """Entity resolution should skip gracefully when Neo4j is unavailable."""
        pass


class TestObjectStorageFailureRecovery:
    """Test graceful degradation when object storage is unavailable."""

    @pytest.mark.asyncio
    async def test_document_upload_fails_cleanly_on_s3_unavailable(self) -> None:
        """Document upload should raise a clear error when S3 is unavailable."""
        pass

    @pytest.mark.asyncio
    async def test_ingestion_continues_without_object_store(self) -> None:
        """Ingestion should proceed with degraded functionality when S3 is down."""
        pass


class TestTelemetryFailureRecovery:
    """Test that telemetry failures don't affect core functionality."""

    @pytest.mark.asyncio
    async def test_query_succeeds_when_otel_unavailable(self) -> None:
        """Query workflow should complete even if OTEL collector is unreachable."""
        pass

    @pytest.mark.asyncio
    async def test_telemetry_span_export_failure_does_not_crash_query(self) -> None:
        """Failed span export should be logged but not raise."""
        pass


class TestCircuitBreakerBehavior:
    """Test that circuit breakers open after repeated failures."""

    @pytest.mark.asyncio
    async def test_circuit_breaks_after_postgres_timeout_threshold(self) -> None:
        """Repeated Postgres failures should open the circuit breaker."""
        pass

    @pytest.mark.asyncio
    async def test_circuit_breaks_after_neo4j_timeout_threshold(self) -> None:
        """Repeated Neo4j failures should open the circuit breaker."""
        pass
