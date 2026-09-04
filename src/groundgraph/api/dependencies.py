"""FastAPI dependencies and request-scoped helpers."""

from __future__ import annotations

import asyncio
import secrets
from urllib.parse import urlparse

from fastapi import Request

from groundgraph.application.health import (
    DependencyHealth,
    HealthReasonCode,
    HealthService,
)
from groundgraph.application.settings import Settings

MAX_REQUEST_ID_LENGTH = 128


def get_request_id(request: Request) -> str | None:
    value = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    if value is None or not 1 <= len(value) <= MAX_REQUEST_ID_LENGTH:
        return None
    return value if value.isascii() and value.replace("-", "").replace("_", "").isalnum() else None


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


class TcpHealthChecker:
    """Check a dependency by opening and promptly closing a TCP connection."""

    async def check(self) -> DependencyHealth:
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            del reader
            writer.close()
            await writer.wait_closed()
        except OSError:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
                details="dependency connection failed",
            )
        return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)

    def __init__(self, name: str, host: str, port: int) -> None:
        self.name = name
        self.host = host
        self.port = port


def build_health_service(settings: Settings) -> HealthService:
    """Build real, bounded connectivity checks for local dependencies."""
    neo4j = urlparse(settings.neo4j_uri)
    s3 = urlparse(settings.s3_endpoint_url)
    return HealthService(
        checkers={
            "postgres": TcpHealthChecker(
                "postgres", settings.postgres_host, settings.postgres_port
            ),
            "neo4j": TcpHealthChecker("neo4j", neo4j.hostname or "localhost", neo4j.port or 7687),
            "minio": TcpHealthChecker("minio", s3.hostname or "localhost", s3.port or 9000),
        }
    )
