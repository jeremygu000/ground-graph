"""Unit tests for API dependency helpers and health checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import SecretStr

from groundgraph.api import dependencies
from groundgraph.application.health import HealthReasonCode
from groundgraph.application.settings import Settings


def test_request_id_from_headers_prefers_valid_header() -> None:
    headers = [(b"x-request-id", b"abc-123")]

    request_id = dependencies.request_id_from_headers(headers)

    assert request_id == "abc-123"


def test_request_id_from_headers_falls_back_for_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "groundgraph.api.dependencies.secrets.token_hex",
        lambda _n: "deadbeefdeadbeefdeadbeef",
    )

    request_id = dependencies.request_id_from_headers([(b"x-request-id", b"no spaces here")])

    assert request_id == "req-deadbeefdeadbeefdeadbeef"


@dataclass
class _Conn:
    executed: list[str]
    closed: int = 0

    async def execute(self, query: str) -> None:
        self.executed.append(query)

    async def close(self) -> None:
        self.closed += 1


class _PgConnect:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def execute(self, query: str) -> None:
        await self._conn.execute(query)

    async def close(self) -> None:
        await self._conn.close()


@pytest.mark.asyncio
async def test_postgres_health_checker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn(executed=[])

    async def _connect(**_kwargs: Any) -> _PgConnect:
        return _PgConnect(conn)

    monkeypatch.setattr("groundgraph.api.dependencies.asyncpg.connect", _connect)

    checker = dependencies.PostgresHealthChecker("localhost", 5432, "user", "pw", "db")
    health = await checker.check()

    assert health.healthy is True
    assert health.reason_code == HealthReasonCode.OK
    assert conn.executed == ["SELECT 1"]


@pytest.mark.asyncio
async def test_postgres_health_checker_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _connect(**_kwargs: Any) -> _PgConnect:
        raise TimeoutError

    monkeypatch.setattr("groundgraph.api.dependencies.asyncpg.connect", _connect)

    checker = dependencies.PostgresHealthChecker("localhost", 5432, "user", "pw", "db")
    health = await checker.check()

    assert health.healthy is False
    assert health.reason_code == HealthReasonCode.TIMEOUT


class _Neo4jSession:
    def __init__(self) -> None:
        self.runs: list[str] = []

    async def __aenter__(self) -> _Neo4jSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def run(self, query: str) -> None:
        self.runs.append(query)


class _Neo4jDriver:
    def __init__(self) -> None:
        self.session_obj = _Neo4jSession()
        self.closed = 0

    def session(self) -> _Neo4jSession:
        return self.session_obj

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_neo4j_health_checker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _Neo4jDriver()

    monkeypatch.setattr(
        "groundgraph.api.dependencies.AsyncGraphDatabase.driver",
        lambda *args, **kwargs: driver,
    )

    checker = dependencies.Neo4jHealthChecker("bolt://localhost:7687", "neo4j", "pw")
    health = await checker.check()

    assert health.healthy is True
    assert driver.session_obj.runs == ["RETURN 1"]
    assert driver.closed == 1


@pytest.mark.asyncio
async def test_minio_health_checker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            self.url = url
            return _Resp()

    monkeypatch.setattr("groundgraph.api.dependencies.httpx.AsyncClient", lambda timeout: _Client())

    checker = dependencies.MinioHealthChecker("http://localhost:9000")
    health = await checker.check()

    assert health.healthy is True


def test_build_health_service_wires_checkers() -> None:
    settings = Settings(
        app_env="test",
        postgres_host="pg",
        postgres_port=5432,
        postgres_user="u",
        postgres_password=SecretStr("p"),
        postgres_db="db",
        neo4j_uri="bolt://neo4j:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("pw"),
        s3_endpoint_url="http://minio:9000",
    )

    service = dependencies.build_health_service(settings)

    assert set(service.checkers) == {"postgres", "neo4j", "minio"}


def test_get_identity_returns_identity_when_headers_present() -> None:
    """get_identity extracts tenant_id and principal from trusted headers."""
    mock_request = object()

    identity = dependencies.get_identity(
        request=mock_request,  # type: ignore[arg-type]
        x_tenant_id="tenant-123",
        x_principal="user-456",
    )

    assert identity.tenant_id == "tenant-123"
    assert identity.principal == "user-456"


def test_get_identity_raises_when_tenant_id_missing() -> None:
    """get_identity raises ValueError when X-Tenant-ID is missing."""
    mock_request = cast(Any, object())

    with pytest.raises(ValueError, match="X-Tenant-ID header is required"):
        dependencies.get_identity(request=mock_request, x_tenant_id=None, x_principal="user-1")


def test_get_identity_raises_when_principal_missing() -> None:
    """get_identity raises ValueError when X-Principal is missing."""
    mock_request = cast(Any, object())

    with pytest.raises(ValueError, match="X-Principal header is required"):
        dependencies.get_identity(request=mock_request, x_tenant_id="tenant-1", x_principal=None)


def test_request_id_from_headers_prefers_request_id_over_correlation_id() -> None:
    """request_id_from_headers prefers x-request-id over x-correlation-id."""
    headers = [(b"x-request-id", b"req-123"), (b"x-correlation-id", b"corr-456")]
    request_id = dependencies.request_id_from_headers(headers)
    assert request_id == "req-123"


def test_request_id_from_headers_rejects_too_long_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """request_id_from_headers rejects request ID longer than 128 chars."""
    monkeypatch.setattr(
        "groundgraph.api.dependencies.secrets.token_hex",
        lambda _n: "deadbeefdeadbeefdeadbeef",
    )
    long_id = "a" * 129
    headers = [(b"x-request-id", long_id.encode())]
    request_id = dependencies.request_id_from_headers(headers)
    assert request_id == "req-deadbeefdeadbeefdeadbeef"


def test_request_id_from_headers_rejects_non_alphanumeric_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """request_id_from_headers rejects request ID with spaces."""
    monkeypatch.setattr(
        "groundgraph.api.dependencies.secrets.token_hex",
        lambda _n: "deadbeefdeadbeefdeadbeef",
    )
    headers = [(b"x-request-id", b"req with spaces")]
    request_id = dependencies.request_id_from_headers(headers)
    assert request_id == "req-deadbeefdeadbeefdeadbeef"


def test_request_id_from_headers_accepts_hyphen_underscore_in_request_id() -> None:
    """request_id_from_headers accepts request ID with hyphens and underscores."""
    headers = [(b"x-request-id", b"req-abc_123")]
    request_id = dependencies.request_id_from_headers(headers)
    assert request_id == "req-abc_123"
