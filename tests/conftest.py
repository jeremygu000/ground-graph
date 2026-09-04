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
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

DOCKER_COMPOSE_FILE = "docker-compose.yml"


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


def _run_compose(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-f", DOCKER_COMPOSE_FILE, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _compose_services() -> list[dict[str, Any]]:
    result = _run_compose(["ps", "--format", "json"])
    if result.returncode != 0:
        return []
    lines = result.stdout.strip().split("\n")
    services = []
    for line in lines:
        if line:
            with suppress(Exception):
                services.append(json.loads(line))
    return services


def _wait_for_stack_healthy(timeout: int = 180) -> StackServices:
    deadline = time.monotonic() + timeout
    last_error: str | None = None

    while time.monotonic() < deadline:
        services = _compose_services()
        if not services:
            time.sleep(2)
            continue

        all_healthy = True
        for svc in services:
            health = svc.get("Health", "")
            state = svc.get("State", "")
            if health in ("healthy", ""):
                continue
            if state == "running" and health == "":
                continue
            all_healthy = False

        if all_healthy and len(services) >= 7:
            break

        try:
            _run_compose(["ps", "--format", "json"], timeout=10)
        except Exception as exc:
            last_error = str(exc)

        time.sleep(3)
    else:
        services = _compose_services()
        raise RuntimeError(
            f"Docker compose services did not become healthy within {timeout}s.\n"
            f"Last error: {last_error}\n"
            f"Running services: {services}"
        )

    return StackServices()


@pytest.fixture(scope="session")
def docker_stack() -> StackServices:
    """Start docker-compose stack and yield connection info; tear down on exit."""
    compose_up = _run_compose(["up", "-d"], timeout=120)
    if compose_up.returncode != 0:
        pytest.skip(
            f"docker compose up failed (may require running Docker daemon):\n"
            f"stdout: {compose_up.stdout}\n"
            f"stderr: {compose_up.stderr}"
        )

    try:
        services = _wait_for_stack_healthy(timeout=180)
        yield services
    finally:
        _run_compose(["down"], timeout=60)


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


async def _stream_app_logs(proc: Any) -> None:
    while True:
        line = proc.stdout.readline()
        if not line:
            break


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
        while True:
            line = await proc.stdout.readline()
            if not line:
                break

    read_task = asyncio.create_task(_read_logs())
    ready = await asyncio.wait_for(_wait_for_app(docker_stack.api_base_url), timeout=45.0)
    if not ready:
        read_task.cancel()
        proc.terminate()
        await proc.wait()
        raise RuntimeError("FastAPI app did not become ready within 45s.")

    try:
        yield []
    finally:
        read_task.cancel()
        with suppress(Exception):
            proc.stdout.close()
        with suppress(Exception):
            proc.kill()
        await asyncio.sleep(0.1)
        with suppress(Exception):
            await proc.wait()
