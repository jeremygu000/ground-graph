"""FastAPI dependencies and request-scoped helpers."""

from __future__ import annotations

import secrets

import asyncpg
import httpx
from neo4j import AsyncGraphDatabase

from groundgraph.application.health import (
    DependencyHealth,
    HealthReasonCode,
    HealthService,
)
from groundgraph.application.settings import Settings

MAX_REQUEST_ID_LENGTH = 128


def request_id_from_headers(headers: list[tuple[bytes, bytes]]) -> str:
    """Return a validated correlation ID from an ASGI header list or create one."""

    values = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in headers
        if key.lower() in {b"x-request-id", b"x-correlation-id"}
    }
    value = values.get("x-request-id") or values.get("x-correlation-id")
    if (
        value
        and 1 <= len(value) <= MAX_REQUEST_ID_LENGTH
        and value.isascii()
        and value.replace("-", "").replace("_", "").isalnum()
    ):
        return value
    return f"req-{secrets.token_hex(12)}"


class PostgresHealthChecker:
    name = "postgres"

    async def check(self) -> DependencyHealth:
        try:
            conn = await asyncpg.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                timeout=2,
            )
            try:
                await conn.execute("SELECT 1")
            finally:
                await conn.close()
            return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
                details="postgres query failed",
            )

    def __init__(self, host: str, port: int, user: str, password: str, database: str) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database


class Neo4jHealthChecker:
    name = "neo4j"

    async def check(self) -> DependencyHealth:
        try:
            driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=1,
                connection_timeout=2.0,
            )
            try:
                async with driver.session() as session:
                    await session.run("RETURN 1")
            finally:
                await driver.close()
            return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
                details="neo4j query failed",
            )

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password


class MinioHealthChecker:
    name = "minio"

    async def check(self) -> DependencyHealth:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._endpoint}/minio/health/ready")
                if resp.status_code == 200:  # noqa: PLR2004
                    return DependencyHealth(
                        name=self.name, healthy=True, reason_code=HealthReasonCode.OK
                    )
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    reason_code=HealthReasonCode.UNHEALTHY,
                    details="minio not ready",
                )
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
                details="minio health check failed",
            )

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint


def build_health_service(settings: Settings) -> HealthService:
    """Build real, bounded connectivity checks for local dependencies."""
    return HealthService(
        checkers={
            "postgres": PostgresHealthChecker(
                settings.postgres_host,
                settings.postgres_port,
                settings.postgres_user,
                settings.postgres_password.get_secret_value(),
                settings.postgres_db,
            ),
            "neo4j": Neo4jHealthChecker(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password.get_secret_value(),
            ),
            "minio": MinioHealthChecker(settings.s3_endpoint_url),
        }
    )
