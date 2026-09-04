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
import uuid
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration


def _run_compose(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "compose", "-f", "docker-compose.yml", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


async def _poll_url(url: str, timeout_sec: float = 30.0, expected_status: int = 200) -> bool:
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
        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"

        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={"x-request-id": f"test-trace-{uuid.uuid4().hex[:12]}"},
                timeout=10.0,
            )
            assert r.status_code == 200

        initial_traces: list[str] = []
        for _ in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{phoenix_url}/v1/projects/default/traces",
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        initial_traces = [t["trace_id"] for t in resp.json().get("data", [])]
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        async with httpx.AsyncClient() as client:
            await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={"x-request-id": f"test-trace-{uuid.uuid4().hex[:12]}"},
                timeout=10.0,
            )

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
                        current_traces = [t["trace_id"] for t in data.get("data", [])]
                        new_traces = [tid for tid in current_traces if tid not in initial_traces]
                        if new_traces:
                            found = True
                            break
            except Exception:
                pass

        assert found, (
            "No new trace appeared in Phoenix after API request. "
            "Verify the OTel Collector is exporting spans to Phoenix and "
            "Phoenix is storing the trace data."
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

        prom_url = f"http://{docker_stack.prometheus_host}:{docker_stack.prometheus_port}"
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{prom_url}/api/v1/query",
                params={"query": "groundgraph_readiness_dependency_healthy"},
                timeout=10.0,
            )
        assert r.status_code == 200, f"Prometheus query API should return 200, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "success", (
            f"Prometheus query failed: {data.get('error', 'unknown')}"
        )
        result = data.get("data", {}).get("result", [])
        assert len(result) > 0, (
            "groundgraph_readiness_dependency_healthy metric should be stored in Prometheus "
            "after being scraped from the collector. This metric appears immediately; "
            "Starlette HTTP metrics require >= 60s to appear (OTel export interval)."
        )


class TestSecretsRedaction:
    """Verify that secrets are redacted from telemetry before export."""

    async def test_authorization_header_is_not_in_server_span(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        secret_token = f"super-secret-token-{uuid.uuid4().hex[:8]}"
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={"Authorization": f"Bearer {secret_token}"},
                timeout=5.0,
            )
        assert r.status_code == 200

        await asyncio.sleep(5)

        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"
        async with httpx.AsyncClient() as client:
            trace_r = await client.get(
                f"{phoenix_url}/v1/projects/default/traces",
                timeout=10.0,
            )

        if trace_r.status_code != 200:
            pytest.fail(
                f"Phoenix trace API should return 200, got {trace_r.status_code}: "
                f"{trace_r.text[:200]}"
            )

        payload = trace_r.json()
        trace_text = str(payload).lower()
        assert secret_token not in trace_text, (
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

    async def test_grafana_dashboard_has_required_panels(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        password = os.environ.get("GRAFANA_PASSWORD", "change-me-local-only")
        async with httpx.AsyncClient(auth=("admin", password)) as client:
            r = await client.get(
                (
                    f"http://{docker_stack.grafana_host}:{docker_stack.grafana_port}"
                    "/api/dashboards/uid/groundgraph-operations"
                ),
                timeout=10.0,
            )
        assert r.status_code == 200, (
            f"Grafana dashboard API should return 200, got {r.status_code}: {r.text[:200]}"
        )
        dashboard = r.json()
        panels_by_title: dict[str, Any] = {}
        for panel in dashboard.get("dashboard", {}).get("panels", []):
            panels_by_title[panel.get("title", "")] = panel

        required_panels = [
            "Request rate (req/s) — groundgraph application",
            "Error rate (5xx / total) — groundgraph application",
            "Latency p50 / p95 / p99 (seconds) — groundgraph application",
        ]
        for title in required_panels:
            assert title in panels_by_title, (
                f"Required panel '{title}' not found in Grafana dashboard. "
                f"Available panels: {list(panels_by_title.keys())}"
            )

        for title in required_panels:
            panel = panels_by_title[title]
            assert panel.get("type") in ("timeseries", "stat", "bargauge"), (
                f"Panel '{title}' should be a timeseries/stat/bargauge, got {panel.get('type')}"
            )
            targets = panel.get("targets", [])
            assert len(targets) > 0, f"Panel '{title}' has no query targets"
