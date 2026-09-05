"""Integration test configuration and shared fixtures.

This module provides a docker-compose stack fixture for tests that require
the full local infrastructure (PostgreSQL/pgvector, Neo4j, MinIO,
OpenTelemetry Collector, Phoenix, Prometheus, Grafana).

The fixture is session-scoped: docker-compose is started once before all
integration tests and torn down after they complete.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

DOCKER_COMPOSE_FILE = "docker-compose.yml"
EXPECTED_SERVICES = frozenset(
    {
        "postgres",
        "neo4j",
        "minio",
        "otel-collector",
        "phoenix",
        "prometheus",
        "grafana",
    }
)
SERVICES_WITH_HEALTHCHECK = frozenset(
    {
        "postgres",
        "neo4j",
        "minio",
        "phoenix",
        "prometheus",
        "grafana",
    }
)


@dataclass
class StackServices:
    """Connection parameters for docker-compose services resolved at startup."""

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    neo4j_host: str = "127.0.0.1"
    neo4j_port: int = 7687
    minio_host: str = "127.0.0.1"
    minio_port: int = 9000
    otel_collector_host: str = "127.0.0.1"
    otel_collector_port: int = 4317
    phoenix_host: str = "127.0.0.1"
    phoenix_port: int = 6006
    prometheus_host: str = "127.0.0.1"
    prometheus_port: int = 9090
    grafana_host: str = "127.0.0.1"
    grafana_port: int = 3001
    api_base_url: str = "http://127.0.0.1:8000"

    health_check_urls: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.health_check_urls = {
            "postgres": f"http://{self.postgres_host}:{self.postgres_port}",
            "neo4j": f"http://{self.neo4j_host}:{self.neo4j_port}",
            "minio": f"http://{self.minio_host}:{self.minio_port}",
            "otel": f"http://{self.otel_collector_host}:13133",
            "phoenix": f"http://{self.phoenix_host}:{self.phoenix_port}/healthz",
            "prometheus": f"http://{self.prometheus_host}:{self.prometheus_port}/-/healthy",
            "grafana": f"http://{self.grafana_host}:{self.grafana_port}/api/health",
        }


def _run_compose(
    args: list[str],
    timeout: int = 60,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-f", DOCKER_COMPOSE_FILE, *args]
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=merged_env,
    )


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _pick_unique_free_port(used_ports: set[int]) -> int:
    while True:
        port = _pick_free_port()
        if port not in used_ports:
            used_ports.add(port)
            return port


def _compose_services(compose_env: dict[str, str]) -> list[dict[str, Any]]:
    result = _run_compose(["ps", "--format", "json"], timeout=60, env=compose_env)
    if result.returncode != 0:
        return []
    lines = result.stdout.strip().split("\n")
    services: list[dict[str, Any]] = []
    for line in lines:
        if line:
            with suppress(Exception):
                services.append(json.loads(line))
    return services


def _wait_for_stack_healthy(
    compose_env: dict[str, str],
    *,
    stack: StackServices,
    timeout: int = 180,
) -> StackServices:
    """Wait until all expected services are running and healthy.

    For services that declare a container healthcheck, both
    ``State == running`` and ``Health == healthy`` are required. For
    services without a healthcheck (e.g. otel-collector), only
    ``State == running`` is required.
    """
    deadline = time.monotonic() + timeout
    last_error: str | None = None

    while time.monotonic() < deadline:
        services = _compose_services(compose_env)
        if not services:
            time.sleep(2)
            continue

        running_names = {svc.get("Service", "") for svc in services}
        missing = EXPECTED_SERVICES - running_names
        if missing:
            time.sleep(2)
            continue

        all_healthy = True
        for svc in services:
            name = svc.get("Service", "")
            if name not in EXPECTED_SERVICES:
                continue
            svc_state = svc.get("State", "")
            if svc_state != "running":
                all_healthy = False
                break
            if name in SERVICES_WITH_HEALTHCHECK:
                health = svc.get("Health", "")
                if health != "healthy":
                    all_healthy = False
                    break

        if all_healthy:
            break

        try:
            _run_compose(["ps", "--format", "json"], timeout=10, env=compose_env)
        except Exception as exc:
            last_error = str(exc)

        time.sleep(3)
    else:
        services = _compose_services(compose_env)
        raise RuntimeError(
            f"Docker compose services did not become healthy within {timeout}s.\n"
            f"Last error: {last_error}\n"
            f"Running services: {services}"
        )

    return stack


@pytest.fixture(scope="session")
def docker_stack() -> Generator[StackServices, None, None]:
    """Start docker-compose stack and yield connection info; tear down on exit.

    Skip policy:
      * Docker CLI binary missing → skip
      * Docker daemon unreachable → skip
    Fail policy:
      * ``docker compose up`` returns non-zero → fail
    """
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available on PATH.")

    docker_info = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if docker_info.returncode != 0:
        pytest.skip(
            f"Docker daemon is not reachable:\n"
            f"stdout: {docker_info.stdout}\n"
            f"stderr: {docker_info.stderr}"
        )

    used_ports: set[int] = set()
    postgres_port = _pick_unique_free_port(used_ports)
    neo4j_http_port = _pick_unique_free_port(used_ports)
    neo4j_bolt_port = _pick_unique_free_port(used_ports)
    minio_port = _pick_unique_free_port(used_ports)
    minio_console_port = _pick_unique_free_port(used_ports)
    otel_grpc_port = _pick_unique_free_port(used_ports)
    otel_http_port = _pick_unique_free_port(used_ports)
    otel_health_port = _pick_unique_free_port(used_ports)
    otel_internal_metrics_port = _pick_unique_free_port(used_ports)
    otel_app_metrics_port = _pick_unique_free_port(used_ports)
    phoenix_port = _pick_unique_free_port(used_ports)
    phoenix_prometheus_port = _pick_unique_free_port(used_ports)
    prometheus_port = _pick_unique_free_port(used_ports)
    grafana_port = _pick_unique_free_port(used_ports)

    compose_env = {
        "POSTGRES_PORT": str(postgres_port),
        "NEO4J_HTTP_PORT": str(neo4j_http_port),
        "NEO4J_BOLT_PORT": str(neo4j_bolt_port),
        "MINIO_PORT": str(minio_port),
        "MINIO_CONSOLE_PORT": str(minio_console_port),
        "OTEL_GRPC_PORT": str(otel_grpc_port),
        "OTEL_HTTP_PORT": str(otel_http_port),
        "OTEL_HEALTH_PORT": str(otel_health_port),
        "OTEL_INTERNAL_METRICS_PORT": str(otel_internal_metrics_port),
        "OTEL_APP_METRICS_PORT": str(otel_app_metrics_port),
        "PHOENIX_PORT": str(phoenix_port),
        "PHOENIX_PROMETHEUS_PORT": str(phoenix_prometheus_port),
        "PROMETHEUS_PORT": str(prometheus_port),
        "GRAFANA_PORT": str(grafana_port),
    }

    stack = StackServices(
        postgres_port=postgres_port,
        neo4j_port=neo4j_bolt_port,
        minio_port=minio_port,
        otel_collector_port=otel_grpc_port,
        phoenix_port=phoenix_port,
        prometheus_port=prometheus_port,
        grafana_port=grafana_port,
    )

    compose_up = _run_compose(["up", "-d"], timeout=120, env=compose_env)
    if compose_up.returncode != 0:
        pytest.fail(
            f"docker compose up failed:\nstdout: {compose_up.stdout}\nstderr: {compose_up.stderr}"
        )

    try:
        services = _wait_for_stack_healthy(
            compose_env,
            stack=stack,
            timeout=180,
        )
        yield services
    finally:
        _run_compose(["down"], timeout=60, env=compose_env)


async def _wait_for_app(base_url: str, timeout_sec: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        await asyncio.sleep(1.5)
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{base_url}/health/live", timeout=2.0)
                if r.status_code == 200:
                    return True
        except Exception:
            pass
    return False


@pytest.fixture(scope="session")
async def app_process(docker_stack: StackServices) -> AsyncGenerator[list[str], None]:
    """Start the FastAPI app connected to the docker-compose stack; yield logs."""
    env = {
        **os.environ,
        "APP_ENV": "development",
        "POSTGRES_HOST": docker_stack.postgres_host,
        "POSTGRES_PORT": str(docker_stack.postgres_port),
        "NEO4J_URI": f"bolt://{docker_stack.neo4j_host}:{docker_stack.neo4j_port}",
        "S3_ENDPOINT_URL": f"http://{docker_stack.minio_host}:{docker_stack.minio_port}",
        "OTEL_EXPORTER_OTLP_ENDPOINT": (
            f"http://{docker_stack.otel_collector_host}:{docker_stack.otel_collector_port}"
        ),
        "OTEL_EXPORTER_OTLP_INSECURE": "true",
        "PHOENIX_COLLECTOR_ENDPOINT": (
            f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"
        ),
        "TELEMETRY_CAPTURE_CONTENT": "false",
        "AUTH_MODE": "local",
        # Integration tests wait for metrics to flow app → collector →
        # Prometheus. The production default is 60s; the test fixture
        # overrides it to 5s so the test_app_metrics_reach_prometheus
        # integration test finishes in seconds rather than minutes.
        "OTEL_METRIC_EXPORT_INTERVAL_MS": "5000",
    }

    cwd = os.getcwd()
    proc = await asyncio.create_subprocess_shell(
        "uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000",
        stdout=asyncio.subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
    )

    async def _read_logs() -> None:
        reader = proc.stdout
        if reader is None:
            return
        while True:
            line = await reader.readline()
            if not line:
                break

    read_task = asyncio.create_task(_read_logs())
    ready = await asyncio.wait_for(_wait_for_app(docker_stack.api_base_url), timeout=45.0)
    if not ready:
        read_task.cancel()
        proc.terminate()
        with suppress(asyncio.CancelledError):
            await read_task
        await proc.wait()
        raise RuntimeError("FastAPI app did not become ready within 45s.")

    try:
        yield []
    finally:
        read_task.cancel()
        with suppress(asyncio.CancelledError):
            await read_task
        proc.terminate()
        with suppress(Exception):
            await proc.wait()
