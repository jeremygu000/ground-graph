"""Tests for health checkers in dependencies."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from groundgraph.api import dependencies
from groundgraph.application.health import HealthReasonCode
from groundgraph.application.settings import Settings


class TestPostgresHealthChecker:
    """Tests for PostgresHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_returns_healthy_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PostgresHealthChecker returns healthy when pg_isready succeeds."""
        checker = dependencies.PostgresHealthChecker(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            database="testdb",
        )

        async def _connect(*args: Any, **kwargs: Any) -> MagicMock:
            conn = MagicMock()
            conn.execute = AsyncMock(return_value=None)
            return conn

        monkeypatch.setattr("groundgraph.api.dependencies.asyncpg.connect", _connect)

        result = await checker.check()

        assert result.healthy is True
        assert result.name == "postgres"
        assert result.reason_code == HealthReasonCode.OK

    @pytest.mark.asyncio
    async def test_check_returns_unhealthy_on_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PostgresHealthChecker returns unhealthy when connection fails."""
        checker = dependencies.PostgresHealthChecker(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            database="testdb",
        )
        monkeypatch.setattr(
            "groundgraph.api.dependencies.asyncpg.connect",
            AsyncMock(side_effect=OSError("connection refused")),
        )

        result = await checker.check()

        assert result.healthy is False
        assert result.reason_code == HealthReasonCode.UNHEALTHY


class TestBuildHealthService:
    """Tests for build_health_service."""

    def test_build_health_service_returns_service_with_all_checkers(self) -> None:
        """build_health_service creates HealthService with postgres, neo4j, and minio checkers."""
        settings = Settings(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_user="test",
            postgres_password=SecretStr("testpassword"),
            postgres_db="testdb",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password=SecretStr("password"),
            s3_endpoint_url="http://localhost:9000",
        )

        service = dependencies.build_health_service(settings)

        assert set(service.checkers) == {"postgres", "neo4j", "minio"}
