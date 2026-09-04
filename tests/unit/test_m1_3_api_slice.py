"""M1.3 API and telemetry slice tests."""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, Response
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import SecretStr

from groundgraph.api.app import create_app
from groundgraph.application.health import (
    DependencyHealth,
    HealthReasonCode,
    HealthService,
)
from groundgraph.application.settings import Settings
from groundgraph.infrastructure.metrics import AppMetrics, init_app_metrics
from groundgraph.infrastructure.telemetry import (
    configure_meter_provider,
    configure_tracing,
    get_meter,
    redact_text,
    sanitize_attributes,
    traced_background_task,
)


def _settings() -> Settings:
    return Settings(
        app_env="test",
        openai_api_key=SecretStr(""),
        otel_exporter_otlp_endpoint="http://localhost:4317",
        otel_exporter_otlp_insecure=True,
        auth_mode="local",
        auth_trusted_headers=False,
    )


def _make_test_metrics() -> tuple[MeterProvider, InMemoryMetricReader, AppMetrics]:
    """Create an in-memory meter provider with OTel instruments for test assertions."""
    reader = InMemoryMetricReader()
    provider = configure_meter_provider(
        "groundgraph",
        None,
        metric_reader=reader,
        enable_otlp=False,
    )
    meter = get_meter(provider, "groundgraph")
    app_metrics = init_app_metrics(meter)
    return provider, reader, app_metrics


class _HealthyChecker:
    def __init__(self, name: str) -> None:
        self.name = name

    async def check(self) -> DependencyHealth:
        return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)


def _healthy_service() -> HealthService:
    return HealthService(
        checkers={name: _HealthyChecker(name) for name in ("postgres", "neo4j", "minio")}
    )


class _UnhealthyChecker:
    name = "neo4j"

    async def check(self) -> DependencyHealth:
        return DependencyHealth(
            name=self.name,
            healthy=False,
            reason_code=HealthReasonCode.UNHEALTHY,
        )


def _unhealthy_service() -> HealthService:
    return HealthService(
        checkers={
            "postgres": _HealthyChecker("postgres"),
            "neo4j": _UnhealthyChecker(),
            "minio": _HealthyChecker("minio"),
        }
    )


async def _request(app: FastAPI, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.get(path, **kwargs)


@pytest.mark.anyio
async def test_create_app_exposes_health_routes() -> None:
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    assert (await _request(app, "/health/live")).status_code == 200
    assert (await _request(app, "/health/ready")).status_code == 200


@pytest.mark.anyio
async def test_create_app_injects_request_ids() -> None:
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    response = await _request(app, "/health/live", headers={"x-request-id": "req-123"})
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-correlation-id"] == "req-123"
    assert response.headers["x-process-time-ms"]
    assert "x-telemetry-redaction-policy" not in response.headers


@pytest.mark.anyio
async def test_invalid_request_id_is_replaced() -> None:
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    response = await _request(app, "/health/live", headers={"x-request-id": "bad id!"})
    assert response.status_code == 200
    assert response.headers["x-request-id"].startswith("req-")
    assert response.headers["x-request-id"] != "bad id!"


@pytest.mark.anyio
async def test_ready_returns_safe_503_for_failed_dependency() -> None:
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_unhealthy_service(), app_metrics=app_metrics)
    response = await _request(app, "/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    failed = next(item for item in body["dependencies"] if item["name"] == "neo4j")
    assert failed["reason_code"] == "unhealthy"
    assert "details" not in failed
    assert "should-not-leak" not in response.text


@pytest.mark.anyio
async def test_ready_does_not_leak_unhandled_probe_exception() -> None:
    class _ExplodingChecker:
        name = "minio"

        async def check(self) -> DependencyHealth:
            raise RuntimeError("postgres://admin:top-secret@internal")

    _mp, _reader, app_metrics = _make_test_metrics()
    service = HealthService(checkers={"minio": _ExplodingChecker()})
    app = create_app(_settings(), health_service=service, app_metrics=app_metrics)
    response = await _request(app, "/health/ready")
    assert response.status_code == 503
    dependency = response.json()["dependencies"][0]
    assert dependency["reason_code"] == "error"
    assert "details" not in dependency
    assert "top-secret" not in response.text


@pytest.mark.anyio
async def test_ready_returns_safe_503_for_dependency_timeout() -> None:
    class _SlowChecker:
        name = "postgres"

        async def check(self) -> DependencyHealth:
            await asyncio.sleep(0.05)
            return DependencyHealth(name=self.name, healthy=True)

    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(
        _settings(),
        health_service=HealthService(checkers={"postgres": _SlowChecker()}, timeout_seconds=0.001),
        app_metrics=app_metrics,
    )
    response = await _request(app, "/health/ready")
    assert response.status_code == 503
    assert response.json()["dependencies"] == [
        {"name": "postgres", "healthy": False, "reason_code": "timeout"}
    ]


@pytest.mark.anyio
async def test_in_memory_exporter_records_safe_server_span() -> None:
    exporter = InMemorySpanExporter()
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(
        _settings(),
        health_service=_healthy_service(),
        span_exporter=exporter,
        telemetry_enabled=False,
        app_metrics=app_metrics,
    )

    @app.get("/probe/{item_id}")
    async def probe(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    response = await _request(
        app,
        "/probe/42",
        headers={"authorization": "Bearer top-secret", "x-request-id": "req-safe"},
    )
    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    server = next(span for span in spans if span.kind.name == "SERVER")
    assert server.attributes is not None
    assert server.attributes["http.request.method"] == "GET"
    assert server.attributes["http.route"] == "/probe/{item_id}"
    assert server.attributes["groundgraph.request_id"] == "req-safe"
    attributes = {str(key).lower(): str(value).lower() for key, value in server.attributes.items()}
    assert all("authorization" not in key for key in attributes)
    assert all("top-secret" not in value for value in attributes.values())


@pytest.mark.anyio
async def test_background_task_child_span_keeps_request_trace() -> None:
    exporter = InMemorySpanExporter()
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(
        _settings(),
        health_service=_healthy_service(),
        span_exporter=exporter,
        telemetry_enabled=False,
        app_metrics=app_metrics,
    )
    completed = asyncio.Event()

    @app.get("/background")
    async def background(background_tasks: BackgroundTasks) -> dict[str, str]:
        callback = traced_background_task(
            app.state.tracer_provider.get_tracer(__name__),
            "background.test",
            completed.set,
        )
        background_tasks.add_task(callback)
        return {"status": "ok"}

    response = await _request(app, "/background")
    assert response.status_code == 200
    await asyncio.sleep(0.1)
    assert completed.is_set()
    spans = exporter.get_finished_spans()
    server = next(span for span in spans if span.name.startswith("GET /background"))
    child = next(span for span in spans if span.name == "background.test")
    assert server.context is not None
    assert child.context is not None
    assert child.context.trace_id == server.context.trace_id


@pytest.mark.anyio
async def test_request_metrics_are_recorded() -> None:
    _mp, reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    await _request(app, "/probe/test")
    collected = reader.get_metrics_data()
    assert collected is not None, "expected metrics data after request"
    assert collected.resource_metrics is not None
    assert len(collected.resource_metrics) >= 1
    rm = collected.resource_metrics[0]
    assert rm.scope_metrics is not None
    assert len(rm.scope_metrics) >= 1
    sm = rm.scope_metrics[0]
    assert len(sm.metrics) >= 1
    metric = sm.metrics[0]
    data_points = metric.data.data_points
    assert len(data_points) >= 1
    assert data_points[0].value >= 1


@pytest.mark.anyio
async def test_server_error_increments_error_counter_and_request_count() -> None:
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)

    @app.get("/returns-error")
    async def returns_error() -> Response:
        return Response(status_code=500)

    response = await _request(app, "/returns-error")
    assert response.status_code == 500


@pytest.mark.anyio
async def test_health_routes_do_not_contribute_to_request_metrics() -> None:
    _mp, reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    await _request(app, "/health/live")
    await _request(app, "/health/ready")
    collected = reader.get_metrics_data()
    if collected is not None and collected.resource_metrics:
        rm = collected.resource_metrics[0]
        if rm.scope_metrics:
            sm = rm.scope_metrics[0]
            request_count_metrics = [m for m in sm.metrics if m.name == "groundgraph.http.requests"]
            if request_count_metrics:
                data_points = request_count_metrics[0].data.data_points
                for dp in data_points:
                    route_attr = getattr(dp.attributes, "get", lambda k, d=None: d)("route", None)
                    assert route_attr != "/health/live"
                    assert route_attr != "/health/ready"


def test_redact_text_replaces_secret_like_terms() -> None:
    assert redact_text("Authorization token secret", ["authorization", "token", "secret"]) == (
        "[REDACTED] [REDACTED] [REDACTED]"
    )


def test_sanitize_attributes_omits_bodies_and_credentials() -> None:
    safe = sanitize_attributes(
        {
            "http.request.method": "GET",
            "authorization": "Bearer secret",
            "x-api-key": "secret-key",
            "request.body": "raw content",
            "api_token": "secret-value",
            "nested": {"not": "scalar"},
        }
    )
    assert safe == {"http.request.method": "GET"}


def test_test_app_never_constructs_an_otlp_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default test configuration must not attempt network telemetry export."""

    def _unexpected_exporter(*_: object, **__: object) -> None:
        pytest.fail("OTLP exporter must not be constructed for a test app")

    monkeypatch.setattr(
        "groundgraph.infrastructure.telemetry.OTLPSpanExporter", _unexpected_exporter
    )
    _mp, _reader, app_metrics = _make_test_metrics()
    app = create_app(_settings(), health_service=_healthy_service(), app_metrics=app_metrics)
    assert app.state.tracer_provider is not None


def test_configure_tracing_rejects_otlp_without_endpoint() -> None:
    """OTLP can only be enabled with an explicit endpoint."""

    with pytest.raises(ValueError, match="otlp_endpoint"):
        configure_tracing("groundgraph", None, enable_otlp=True)


def test_repeated_test_app_factories_do_not_change_global_provider() -> None:
    """App-local providers avoid global-provider warnings and state mutation."""

    global_provider = trace.get_tracer_provider()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _mp1, _r1, am1 = _make_test_metrics()
        first = create_app(_settings(), health_service=_healthy_service(), app_metrics=am1)
        _mp2, _r2, am2 = _make_test_metrics()
        second = create_app(_settings(), health_service=_healthy_service(), app_metrics=am2)
    assert first.state.tracer_provider is not second.state.tracer_provider
    assert trace.get_tracer_provider() is global_provider
    assert not captured
