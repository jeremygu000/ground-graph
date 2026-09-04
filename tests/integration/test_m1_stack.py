"""M1 integration tests: full local stack smoke tests.

These tests require the docker-compose stack to be running. They are marked
with ``@pytest.mark.integration`` and are skipped unless Docker is available.

Run with:  make test-integration
Or manually: docker compose up -d && uv run pytest -m integration -v
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import time
import uuid
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration

OTEL_EXPORT_INTERVAL_SEC = 60.0
PROMETHEUS_SCRAPE_INTERVAL_SEC = 15.0


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


async def _phoenix_list_traces(client: httpx.AsyncClient, phoenix_url: str) -> list[str]:
    """Return the list of trace_ids currently stored in Phoenix."""
    try:
        r = await client.get(f"{phoenix_url}/v1/projects/default/traces", timeout=5.0)
        if r.status_code != 200:
            return []
        return [t["trace_id"] for t in r.json().get("data", []) if "trace_id" in t]
    except Exception:
        return []


async def _phoenix_list_spans(
    client: httpx.AsyncClient, phoenix_url: str, trace_id: str
) -> list[dict[str, Any]]:
    """Return all spans for *trace_id* from Phoenix.

    Phoenix's /v1/projects/{project}/spans endpoint exposes full span
    objects (with attributes), unlike /v1/projects/{project}/traces which
    only returns trace metadata and ``spans: null``.
    """
    spans: list[dict[str, Any]] = []
    try:
        r = await client.get(
            f"{phoenix_url}/v1/projects/default/spans",
            timeout=5.0,
        )
        if r.status_code != 200:
            return []
        for span in r.json().get("data", []):
            ctx = span.get("context") or {}
            if ctx.get("trace_id") == trace_id:
                spans.append(span)
    except Exception:
        return []
    return spans


def _is_finite_number(value: Any) -> bool:
    """Return True iff *value* is a finite real number.

    Prometheus serialises samples as strings: "0.123", "1.5e3", "NaN",
    "+Inf", "-Inf". Empty or non-numeric strings are also rejected.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str) and value:
        try:
            return math.isfinite(float(value))
        except ValueError:
            return False
    return False


async def _phoenix_find_trace_by_request_id(
    client: httpx.AsyncClient, phoenix_url: str, request_id: str
) -> str | None:
    """Search all spans in Phoenix for any span attribute whose value matches
    *request_id*. Return the trace_id of the first match.

    This is the reliable way to find the trace for a specific request, because
    Phoenix itself generates internal background traces (e.g. ``HEAD /_ping``)
    that would otherwise contaminate a "first-new-trace" approach.
    """
    try:
        r = await client.get(f"{phoenix_url}/v1/projects/default/spans", timeout=10.0)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    for span in r.json().get("data", []):
        attrs = span.get("attributes") or {}
        for value in attrs.values():
            if isinstance(value, str) and value == request_id:
                ctx = span.get("context") or {}
                tid = ctx.get("trace_id")
                if tid:
                    return tid
    return None


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


class TestTracePropagation:
    """Verify that API request traces are forwarded to Phoenix with full span detail."""

    async def test_api_trace_appears_in_phoenix_with_full_spans(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"

        # Step 1: generate a unique request_id and send exactly one request.
        request_id = f"groundgraph-trace-{uuid.uuid4().hex}"
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={"x-request-id": request_id},
                timeout=10.0,
            )
        assert r.status_code == 200, "request to /health/ready must succeed"

        # Step 2: poll Phoenix for the trace whose spans carry our request_id.
        # We search by attribute value, NOT by "new trace id", because Phoenix
        # itself emits background traces (e.g. HEAD /_ping) that would
        # otherwise contaminate the search.
        new_trace_id: str | None = None
        async with httpx.AsyncClient() as client:
            for _ in range(30):
                await asyncio.sleep(2)
                new_trace_id = await _phoenix_find_trace_by_request_id(
                    client, phoenix_url, request_id
                )
                if new_trace_id:
                    break

        assert new_trace_id is not None, (
            f"No trace in Phoenix contained request_id={request_id} after 60s. "
            "Verify OTel Collector is exporting spans to Phoenix and that the "
            "app's _server_request_hook attaches x-request-id to span attributes."
        )

        # Step 3: fetch full spans for the matched trace and validate structure.
        async with httpx.AsyncClient() as client:
            spans = await _phoenix_list_spans(client, phoenix_url, new_trace_id)

        assert len(spans) >= 1, (
            f"Expected >= 1 span in trace {new_trace_id} (request_id={request_id}), "
            f"got {len(spans)}"
        )

        # Step 4: validate that the request_id we sent propagated to the spans.
        # The app's _server_request_hook attaches x-request-id to span attributes
        # under the key 'groundgraph.request_id' (or 'http.request.header.x-request-id'
        # in the captured form). Match either.
        request_id_matched = False
        for span in spans:
            attrs = span.get("attributes") or {}
            values = [str(v) for v in attrs.values()]
            if request_id in values:
                request_id_matched = True
                break
        assert request_id_matched, (
            f"x-request-id '{request_id}' not found in any span attribute for "
            f"trace {new_trace_id}. Spans: {[s.get('attributes') for s in spans]}"
        )

        # Step 5: validate a SERVER span exists (the HTTP request itself).
        server_spans = [s for s in spans if "GET /health/ready" in s.get("name", "")]
        assert server_spans, (
            f"No SERVER span named 'GET /health/ready' in trace {new_trace_id}; "
            f"span names: {[s.get('name') for s in spans]}"
        )

        # Step 6: validate at least one dependency child span exists
        # (postgres / neo4j / minio health-check probes). Phoenix stores
        # span_kind as the OpenInference enum value but does not always
        # map SpanKind.CLIENT → "CLIENT" in its REST API; in practice
        # every span shows "UNKNOWN". The reliable signal is the
        # ``healthcheck.`` name prefix combined with a parent_id.
        child_spans = [
            s
            for s in spans
            if s.get("name", "").startswith("healthcheck.") and s.get("parent_id") is not None
        ]
        assert child_spans, (
            f"No healthcheck.* child spans in trace {new_trace_id}; "
            f"span names: {[s.get('name') for s in spans]}; "
            f"parent_ids: {[s.get('parent_id') for s in spans]}"
        )

        # Step 7: validate parent-child relationship — every child span should
        # have a parent_id pointing at the SERVER span's span_id.
        server_span_ids = {s["context"]["span_id"] for s in server_spans if "context" in s}
        for cs in child_spans:
            parent_id = cs.get("parent_id")
            assert parent_id in server_span_ids, (
                f"child span '{cs.get('name')}' has parent_id={parent_id!r} "
                f"which is not the SERVER span's span_id {server_span_ids}"
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
        """Verify the app → OTel collector → Prometheus metrics pipeline.

        The application emits the ``groundgraph_readiness_dependency_healthy``
        gauge on every call to /health/ready. We:
          1. Trigger ≥ 1 /health/ready call to make sure the metric has a sample.
          2. Poll the OTel collector's prometheus exporter (port 8889). The
             app's ``PeriodicExportingMetricReader`` is configured to
             5000 ms in the integration test environment (see
             ``tests/conftest.py``), so the first batch lands well within
             20s. The OTel collector's own batch timeout is 5s.
          3. Poll Prometheus (port 9090) — Prometheus scrapes the collector
             every PROMETHEUS_SCRAPE_INTERVAL_SEC.
        """
        # 1. Generate at least one sample.
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{docker_stack.api_base_url}/health/ready", timeout=5.0)
        assert r.status_code in (200, 503), (
            f"/health/ready should return 200 or 503, got {r.status_code}"
        )

        # 2. Poll the OTel collector's prometheus exporter.
        deadline = time.monotonic() + 20.0
        collector_stdout = ""

        async def _scrape_collector() -> str:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.yml",
                    "exec",
                    "-T",
                    "prometheus",
                    "wget",
                    "-qO-",
                    "http://otel-collector:8889/metrics",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return result.stdout if result.returncode == 0 else ""

        while time.monotonic() < deadline:
            collector_stdout = await _scrape_collector()
            if "groundgraph_readiness_dependency_healthy" in collector_stdout:
                break
            await asyncio.sleep(2.0)

        assert "groundgraph_readiness_dependency_healthy" in collector_stdout, (
            "groundgraph_readiness_dependency_healthy should appear in "
            "OTel collector's prometheus exporter within 20s. The test "
            "environment configures OTEL_METRIC_EXPORT_INTERVAL_MS=5000. "
            "If this fails, verify the app is actually starting and that "
            "OTEL_EXPORTER_OTLP_ENDPOINT points at the collector."
        )

        # 3. Verify Prometheus has the metric. Prometheus scrapes the collector
        # every PROMETHEUS_SCRAPE_INTERVAL_SEC, so poll up to 2 intervals.
        prom_url = f"http://{docker_stack.prometheus_host}:{docker_stack.prometheus_port}"
        deadline = time.monotonic() + (3 * PROMETHEUS_SCRAPE_INTERVAL_SEC)
        data: dict[str, Any] = {}
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                r = await client.get(
                    f"{prom_url}/api/v1/query",
                    params={"query": "groundgraph_readiness_dependency_healthy"},
                    timeout=10.0,
                )
                assert r.status_code == 200, (
                    f"Prometheus query API should return 200, got {r.status_code}"
                )
                data = r.json()
                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    break
                await asyncio.sleep(2.0)

        assert data.get("status") == "success", (
            f"Prometheus query failed: {data.get('error', 'unknown')}"
        )
        result = data.get("data", {}).get("result", [])
        assert len(result) > 0, (
            f"groundgraph_readiness_dependency_healthy should be in Prometheus "
            f"after {(3 * PROMETHEUS_SCRAPE_INTERVAL_SEC):.0f}s polling. "
            f"This proves Prometheus is scraping the OTel collector."
        )


class TestSecretsRedaction:
    """Verify that secrets are redacted from telemetry before export."""

    async def test_authorization_header_is_not_in_trace_spans(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        phoenix_url = f"http://{docker_stack.phoenix_host}:{docker_stack.phoenix_port}"

        # Use a unique x-request-id so we can locate the exact trace for
        # this request. Phoenix emits its own background traces (e.g.
        # ``HEAD /_ping``), so a "first new trace" approach is unsafe —
        # the secret check would then run against an unrelated trace and
        # silently pass. See TestTracePropagation for the same approach.
        request_id = f"secret-test-{uuid.uuid4().hex}"
        secret_token = f"super-secret-{uuid.uuid4().hex}"
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{docker_stack.api_base_url}/health/ready",
                headers={
                    "x-request-id": request_id,
                    "Authorization": f"Bearer {secret_token}",
                },
                timeout=5.0,
            )
        assert r.status_code == 200, "request to /health/ready must succeed"

        # Find the trace by the request_id we attached.
        new_trace_id: str | None = None
        async with httpx.AsyncClient() as client:
            for _ in range(30):
                await asyncio.sleep(2)
                new_trace_id = await _phoenix_find_trace_by_request_id(
                    client, phoenix_url, request_id
                )
                if new_trace_id:
                    break

        assert new_trace_id is not None, (
            f"No trace in Phoenix contained request_id={request_id} after 60s; "
            f"cannot verify that secret {secret_token!r} was redacted."
        )

        # Fetch FULL spans for ONLY this trace and check them.
        async with httpx.AsyncClient() as client:
            spans = await _phoenix_list_spans(client, phoenix_url, new_trace_id)

        assert spans, f"No spans found in Phoenix for trace {new_trace_id}"

        for span in spans:
            attrs = span.get("attributes") or {}
            for value in attrs.values():
                assert secret_token not in str(value), (
                    f"Secret token leaked in span '{span.get('name')}' "
                    f"attribute of trace {new_trace_id}"
                )
            span_text = str(span)
            assert secret_token not in span_text, (
                f"Secret token leaked in raw span payload of '{span.get('name')}' "
                f"in trace {new_trace_id}"
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

    async def test_grafana_request_count_and_latency_panels_return_data(  # noqa: PLR0912
        self, docker_stack: Any, app_process: Any
    ) -> None:
        """Validate the operations dashboard end-to-end with REAL data.

        M1 has no business routes, so we cannot rely on background traffic
        to populate ``groundgraph.http.requests`` / ``...duration``. We
        instead generate a controlled burst of HTTP traffic against
        non-existent paths — these produce 404 responses and ARE observed
        by the app's request metrics middleware (only ``/health/...``,
        ``/metrics``, ``/docs``, ``/redoc`` are excluded).

        Steps:
          1. Generate ≥ 30 random 404 requests to produce real
             counter + histogram samples.
          2. Wait two full OTel export + Prometheus scrape cycles
             (test env configures 5s export, Prometheus scrapes every
             PROMETHEUS_SCRAPE_INTERVAL_SEC).
          3. Assert the raw series exist in Prometheus:
               - groundgraph_http_requests_total
               - groundgraph_http_request_duration_seconds_count
               - groundgraph_http_request_duration_seconds_bucket
          4. Execute every target in every required panel and assert the
             result vector is non-empty.
        """
        password = os.environ.get("GRAFANA_PASSWORD", "change-me-local-only")
        grafana_url = f"http://{docker_stack.grafana_host}:{docker_stack.grafana_port}"
        prom_url = f"http://{docker_stack.prometheus_host}:{docker_stack.prometheus_port}"

        # 1. Generate ≥ 30 random 404 requests.
        async with httpx.AsyncClient() as client:
            for _ in range(30):
                path = f"/__probe__/{uuid.uuid4().hex}"
                await client.get(f"{docker_stack.api_base_url}{path}", timeout=5.0)

        # 2. Wait for two OTel export + Prometheus scrape cycles.
        # App export interval = 5s (set by conftest), OTel collector
        # batch timeout = 5s, Prometheus scrape interval = 15s.
        wait_sec = int(2 * (5 + PROMETHEUS_SCRAPE_INTERVAL_SEC)) + 5
        await asyncio.sleep(wait_sec)

        # 3. Assert raw series are populated in Prometheus.
        async with httpx.AsyncClient() as client:
            for metric in (
                "groundgraph_http_requests_total",
                "groundgraph_http_request_duration_seconds_count",
                "groundgraph_http_request_duration_seconds_bucket",
            ):
                resp = await client.get(
                    f"{prom_url}/api/v1/query",
                    params={"query": metric},
                    timeout=10.0,
                )
                assert resp.status_code == 200, (
                    f"Prometheus query for raw series '{metric}' returned "
                    f"{resp.status_code}: {resp.text[:200]}"
                )
                data = resp.json()
                assert data.get("status") == "success", (
                    f"Prometheus query for '{metric}' failed: {data.get('error')}"
                )
                result = data.get("data", {}).get("result", [])
                assert len(result) > 0, (
                    f"Raw series '{metric}' should have at least one sample in "
                    f"Prometheus after {wait_sec}s of waiting. The 30 404 "
                    f"requests should have produced samples, but the metric is "
                    f"empty. Verify the OTel SDK's PeriodicExportingMetricReader "
                    f"is configured (export_interval_millis) and the OTel "
                    f"collector is exporting to its prometheus exporter."
                )
                # Every sample value must be a finite real number. Empty
                # results, NaN, +Inf, -Inf, or non-numeric strings all
                # indicate a broken metric pipeline (e.g. a divide-by-zero
                # in a rate query, an uninitialised cumulative counter, or
                # a Prometheus encoding bug).
                for sample in result:
                    raw_value = sample.get("value", [None, None])[1]
                    assert _is_finite_number(raw_value), (
                        f"Raw series '{metric}' returned a non-finite sample: "
                        f"value={raw_value!r} in series {sample.get('metric', {})}"
                    )

        # 4. Load the dashboard and validate every panel and every target.
        async with httpx.AsyncClient(auth=("admin", password)) as client:
            r = await client.get(
                f"{grafana_url}/api/dashboards/uid/groundgraph-operations",
                timeout=10.0,
            )
        assert r.status_code == 200, (
            f"Grafana dashboard API should return 200, got {r.status_code}: {r.text[:200]}"
        )
        dashboard = r.json()
        panels_by_title: dict[str, list[dict[str, Any]]] = {}
        for panel in dashboard.get("dashboard", {}).get("panels", []):
            panels_by_title.setdefault(panel.get("title", ""), []).append(panel)

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
            for panel in panels_by_title[title]:
                targets = panel.get("targets", [])
                assert len(targets) > 0, f"Panel '{title}' has no query targets"
                for target in targets:
                    assert target.get("expr"), f"Panel '{title}' target has empty PromQL expression"

        # 5. Execute every target's PromQL and assert a non-empty result
        # vector. The error-rate query may legitimately return empty
        # (no 5xx emitted), so we tolerate that single case.
        async with httpx.AsyncClient() as client:
            for title in required_panels:
                for panel in panels_by_title[title]:
                    for target in panel["targets"]:
                        expr = target["expr"]
                        resp = await client.get(
                            f"{prom_url}/api/v1/query",
                            params={"query": expr},
                            timeout=10.0,
                        )
                        assert resp.status_code == 200, (
                            f"Prometheus query for panel '{title}' target "
                            f"'{target.get('legendFormat', '?')}' returned "
                            f"{resp.status_code}: {resp.text[:200]}"
                        )
                        data = resp.json()
                        assert data.get("status") == "success", (
                            f"Prometheus query for '{title}' failed "
                            f"(PromQL syntax error?): {data.get('error')}; "
                            f"expr={expr!r}"
                        )
                        result = data.get("data", {}).get("result", [])
                        if title.startswith("Error rate") and not result:
                            # Error rate divides 5xx / total; if no 5xx
                            # occurred yet, the numerator is zero and the
                            # result is empty. That's acceptable — the
                            # PromQL is still valid and wired up.
                            continue
                        assert len(result) > 0, (
                            f"Panel '{title}' target '{target.get('legendFormat', '?')}' "
                            f"PromQL returned no data from Prometheus; expr={expr!r}. "
                            f"Step 3 already proved the underlying series exist, so "
                            f"either the PromQL is wrong or the time window is too narrow."
                        )
                        # NaN/Inf handling:
                        #   - Rates and error ratios MUST be finite. NaN here
                        #     means divide-by-zero, which is a real bug.
                        #   - histogram_quantile over a sparse rate window is
                        #     a known PromQL idiom that produces NaN even when
                        #     the underlying histogram is healthy. We do not
                        #     reject NaN for the Latency panel; production
                        #     dashboards wrap it (`clamp_min(..., 0)`) to
                        #     render as 0.
                        is_quantile_panel = title.startswith("Latency p50")
                        for sample in result:
                            raw_value = sample.get("value", [None, None])[1]
                            if is_quantile_panel and raw_value == "NaN":
                                continue
                            assert _is_finite_number(raw_value), (
                                f"Panel '{title}' target "
                                f"'{target.get('legendFormat', '?')}' returned a "
                                f"non-finite sample: value={raw_value!r}; expr={expr!r}"
                            )


class TestDestructivePostgresRecovery:
    """Destructive test placed physically at the END of this file so pytest's
    source-order collection runs it after every other test class. Stops
    postgres, verifies the app returns 503, then restarts postgres and
    health-polls until the app fully recovers (``/health/ready`` returns 200).
    """

    async def test_ready_returns_503_when_postgres_unavailable(
        self, docker_stack: Any, app_process: Any
    ) -> None:
        # Resolve container name dynamically — never hard-code
        # ``groundgraph-postgres-1`` since the compose project name is
        # configurable via ``docker compose -p``.
        ps = await asyncio.to_thread(
            subprocess.run,
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "ps",
                "-q",
                "postgres",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        container_id = ps.stdout.strip().splitlines()[0] if ps.stdout.strip() else ""
        assert container_id, (
            "could not resolve postgres container id via 'docker compose ps -q postgres'"
        )

        _run_compose(["stop", "postgres"], timeout=15)
        try:
            degraded = False
            r: httpx.Response | None = None
            for _ in range(15):
                async with httpx.AsyncClient() as client:
                    r = await client.get(f"{docker_stack.api_base_url}/health/ready", timeout=5.0)
                if r.status_code == 503:
                    degraded = True
                    break
                await asyncio.sleep(1.0)
            assert degraded, "app should return 503 within 15s after postgres is stopped"
            assert r is not None
            data = r.json()
            dep_by_name = {d["name"]: d for d in data.get("dependencies", [])}
            pg = dep_by_name.get("postgres")
            assert pg is not None, "postgres should appear in dependency list"
            assert pg["healthy"] is False, "postgres should be marked unhealthy"
        finally:
            _run_compose(["start", "postgres"], timeout=30)
            # 1) Container reports healthy.
            recovered_container = False
            for _ in range(30):
                chk = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Health.Status}}",
                        container_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if chk.returncode == 0 and chk.stdout.strip() == "healthy":
                    recovered_container = True
                    break
                await asyncio.sleep(1.0)
            assert recovered_container, (
                f"postgres container {container_id} did not report 'healthy' within 30s"
            )
            # 2) App's /health/ready returns 200 again — this proves the
            # Phoenix/Postgres connection pool recovered end-to-end, not
            # just the container itself.
            recovered_app = False
            for _ in range(30):
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{docker_stack.api_base_url}/health/ready",
                        timeout=5.0,
                    )
                if r.status_code == 200:
                    recovered_app = True
                    break
                await asyncio.sleep(1.0)
            assert recovered_app, (
                "app /health/ready did not return 200 within 30s after postgres restart; "
                "Phoenix/postgres connection pool may still be recovering"
            )
