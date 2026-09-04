"""M1 integration tests: full local stack smoke tests.

These tests require the docker-compose stack to be running. They are marked
with ``@pytest.mark.integration`` and are skipped unless Docker is available.

Run with:  make test-integration
Or manually: docker compose up -d && uv run pytest -m integration -v
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration


def _run_compose(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-f", "docker-compose.yml", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


async def _poll_url(url: str, timeout_sec: float = 30.0, expected_status: int = 200) -> bool:
    """Poll a URL until it returns expected_status or the timeout expires."""
    deadline = time.monotonic() + timeout_sec
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            await asyncio.sleep(1.5)
            try:
                r = await client.get(url, timeout=3.0)
                if r.status_code == expected_status:
                    return True
            except Exception:
                pass
    return False


class TestDockerComposeValidation:
    """Verify docker-compose configuration and service discovery."""

    def test_compose_config_is_valid(self, docker_stack: Any) -> None:
        result = _run_compose(["config", "-q"])
        assert result.returncode == 0, f"compose config failed: {result.stderr}"

    def test_all_services_registered(self, docker_stack: Any) -> None:
        result = _run_compose(["ps", "--services"])
        assert result.returncode == 0
        services = set(result.stdout.strip().split("\n"))
        long_running = {
            "postgres",
            "neo4j",
            "minio",
            "otel-collector",
            "phoenix",
            "prometheus",
            "grafana",
        }
        assert long_running.issubset(services), (
            f"Expected long-running services not found in compose. "
            f"Expected: {long_running}, got: {services}"
        )


class TestHealthEndpoints:
    """Verify health/readiness endpoints report correct dependency status."""

    async def test_live_endpoint_returns_200(self, docker_stack: Any, app_process: Any) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{docker_stack.api_base_url}/health/live", timeout=5.0)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"

    async def test_ready_endpoint_reports_all_dependencies(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{docker_stack.api_base_url}/health/ready", timeout=10.0)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")
        deps = {d["name"] for d in data.get("dependencies", [])}
        assert "postgres" in deps, "postgres must be in dependency list"
        assert "neo4j" in deps, "neo4j must be in dependency list"
        assert "minio" in deps, "minio must be in dependency list"

    async def test_ready_returns_503_when_postgres_unavailable(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        _run_compose(["stop", "postgres"], timeout=15)
        try:
            await asyncio.sleep(5)
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{docker_stack.api_base_url}/health/ready", timeout=10.0)
            assert r.status_code == 503, "ready should return 503 when postgres is down"
            data = r.json()
            dep_by_name = {d["name"]: d for d in data.get("dependencies", [])}
            pg = dep_by_name.get("postgres")
            assert pg is not None, "postgres should appear in dependency list"
            assert pg["healthy"] is False, "postgres should be marked unhealthy"
        finally:
            _run_compose(["start", "postgres"], timeout=30)
            await asyncio.sleep(8)


class TestTracePropagation:
    """Verify that API request traces are forwarded to Phoenix via the OTel Collector."""

    async def test_api_trace_appears_in_phoenix(self, docker_stack: Any, app_process: Any) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={"x-request-id": "test-trace-123"},
                timeout=10.0,
            )
            assert r.status_code == 200

        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"
        found = False
        for _ in range(20):
            await asyncio.sleep(3)
            try:
                async with httpx.AsyncClient() as client:
                    trace_r = await client.get(
                        f"{phoenix_url}/v1/projects/default/traces",
                        timeout=5.0,
                    )
                    if trace_r.status_code == 200:
                        data = trace_r.json()
                        if len(data.get("data", [])) > 0:
                            found = True
                            break
            except Exception:
                pass

        assert found, (
            "No traces found in Phoenix for service 'groundgraph' after 60s. "
            "Verify that OTel Collector is exporting spans to Phoenix."
        )


class TestPrometheusMetrics:
    """Verify that the Prometheus scrape endpoint exposes application metrics."""

    async def test_otel_collector_prometheus_exporter_is_scrapable(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        result = _run_compose(
            [
                "exec",
                "-T",
                "prometheus",
                "wget",
                "-qO-",
                "http://otel-collector:8888/metrics",
            ],
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Prometheus failed to scrape otel-collector internal metrics: {result.stderr}"
        )
        assert "otelcol_" in result.stdout, (
            "otelcol_ metrics should be present in scrape output (collector internal telemetry)"
        )

    async def test_app_metrics_reach_prometheus(self, docker_stack: Any, app_process: Any) -> None:
        async with httpx.AsyncClient() as client:
            await client.get(f"{docker_stack.api_base_url}/health/ready", timeout=5.0)
            await asyncio.sleep(2)

        result = _run_compose(
            [
                "exec",
                "-T",
                "prometheus",
                "wget",
                "-qO-",
                "http://otel-collector:8889/metrics",
            ],
            timeout=15,
        )
        assert result.returncode == 0
        output = result.stdout
        assert "groundgraph_readiness_dependency_healthy" in output, (
            "groundgraph_readiness_dependency_healthy metric should be "
            "present in collector scrape output"
        )


class TestSecretsRedaction:
    """Verify that secrets are redacted from telemetry before export."""

    async def test_authorization_header_is_not_in_server_span(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/live",
                headers={"Authorization": "Bearer super-secret-token-xyz"},
                timeout=5.0,
            )
        assert r.status_code == 200

        await asyncio.sleep(5)

        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"
        async with httpx.AsyncClient() as client:
            try:
                trace_r = await client.get(
                    f"{phoenix_url}/v1/projects/default/traces",
                    timeout=10.0,
                )
            except Exception as exc:
                pytest.fail(f"Phoenix trace API should be reachable: {exc}")

        if trace_r.status_code == 200:
            payload = trace_r.json()
            trace_text = str(payload).lower()
            assert "super-secret-token-xyz" not in trace_text, (
                "Raw secret token must not appear in Phoenix trace data"
            )
            assert "bearer" not in trace_text or "redacted" in trace_text, (
                "Authorization header value must be redacted from telemetry"
            )


class TestGrafanaObservability:
    """Verify Grafana dashboards are accessible and data sources are configured."""

    async def test_grafana_is_reachable(self, docker_stack: Any, app_process: Any) -> None:
        url = f"http://{docker_stack.grafana_host}:{docker_stack.grafana_port}/api/health"
        reachable = await _poll_url(url, timeout_sec=30.0)
        assert reachable, f"Grafana health endpoint did not become reachable: {url}"

    async def test_grafana_prometheus_datasource_is_configured(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        password = os.environ.get("GRAFANA_PASSWORD", "change-me-local-only")
        async with httpx.AsyncClient(auth=("admin", password)) as client:
            r = await client.get(
                (f"http://{docker_stack.grafana_host}:{docker_stack.grafana_port}/api/datasources"),
                timeout=10.0,
            )
        assert r.status_code == 200, "Grafana datasources API should return 200"
        datasources = r.json()
        names = [ds.get("name", "").lower() for ds in datasources]
        assert "prometheus" in names, (
            f"Prometheus datasource should be configured in Grafana, got: {names}"
        )
