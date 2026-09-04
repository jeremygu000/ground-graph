"""FastAPI composition root for the M1.3 observability slice."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span

from groundgraph.api.dependencies import build_health_service, request_id_from_headers
from groundgraph.api.health import router as health_router
from groundgraph.application.health import HealthService
from groundgraph.application.settings import Settings, get_settings
from groundgraph.infrastructure.logging import (
    bind_request_context,
    clear_request_context,
    configure_json_logging,
)
from groundgraph.infrastructure.metrics import AppMetrics, init_app_metrics
from groundgraph.infrastructure.telemetry import (
    configure_meter_provider,
    configure_tracing,
    get_meter,
    sanitize_attributes,
    shutdown_meter_provider,
    shutdown_tracing,
)

SERVER_ERROR_STATUS = 500

_ROUTE_EXCLUDE_METRICS = frozenset(
    {
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/redoc",
    }
)


def _is_excluded_route(route: str) -> bool:
    return route in _ROUTE_EXCLUDE_METRICS


def _safe_route(request: Request) -> str:
    """Return route path from the matched route, or '__unmatched__' if no route matched."""
    route_obj = request.scope.get("route")
    if route_obj is None:
        return "__unmatched__"
    path = getattr(route_obj, "path", None)
    return path if isinstance(path, str) else "__unmatched__"


def _server_request_hook(span: Span, scope: dict[str, object]) -> None:
    """Attach safe, stable request metadata to the instrumented server span."""

    raw_headers = scope.get("headers", [])
    headers = (
        [
            (key, value)
            for key, value in raw_headers
            if isinstance(key, bytes) and isinstance(value, bytes)
        ]
        if isinstance(raw_headers, list)
        else []
    )
    request_id = request_id_from_headers(headers)
    state = scope.get("state")
    if not isinstance(state, dict):
        state = {}
        scope["state"] = state
    state["groundgraph_request_id"] = request_id
    state["groundgraph_server_span"] = span
    for key, value in sanitize_attributes(
        {
            "groundgraph.request_id": request_id,
            "groundgraph.correlation_id": request_id,
            "http.request.method": str(scope.get("method", "")),
        }
    ).items():
        span.set_attribute(key, value)


def _log_request(
    event: str,
    *,
    method: str,
    route: str,
    status_code: int | None = None,
    duration_seconds: float | None = None,
    error_type: str | None = None,
) -> None:
    fields: dict[str, str | int | float] = {"method": method, "route": route}
    if status_code is not None:
        fields["status_code"] = status_code
    if duration_seconds is not None:
        fields["duration_ms"] = round(duration_seconds * 1000, 3)
    if error_type is not None:
        fields["error_type"] = error_type
    logger = structlog.get_logger(__name__)
    if status_code is not None and status_code >= SERVER_ERROR_STATUS:
        logger.error(event, **fields)
    else:
        logger.info(event, **fields)


def create_app(  # noqa: PLR0915 - composition root keeps app lifecycle wiring together
    settings: Settings | None = None,
    *,
    health_service: HealthService | None = None,
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    telemetry_enabled: bool | None = None,
    app_metrics: AppMetrics | None = None,
) -> FastAPI:
    """Create the API application.

    The app is test-friendly: tracing and metrics can be disabled for unit tests,
    and the process-wide providers are only set once.
    """

    settings = settings or get_settings()
    configure_json_logging(settings.log_level)
    tracing_enabled = not settings.is_test if telemetry_enabled is None else telemetry_enabled
    tracer_provider = configure_tracing(
        settings.otel_service_name,
        settings.otel_exporter_otlp_endpoint,
        exporter=span_exporter,
        enable_otlp=tracing_enabled,
        otlp_insecure=settings.otel_exporter_otlp_insecure,
    )
    meter_provider = configure_meter_provider(
        settings.otel_service_name,
        settings.otel_exporter_otlp_endpoint,
        metric_reader=metric_reader,
        enable_otlp=tracing_enabled,
        otlp_insecure=settings.otel_exporter_otlp_insecure,
        export_interval_millis=settings.otel_metric_export_interval_ms,
    )

    if app_metrics is None:
        meter = get_meter(meter_provider, settings.otel_service_name)
        app_metrics = init_app_metrics(meter)
    else:
        meter = get_meter(meter_provider, settings.otel_service_name)

    instrumented = tracing_enabled or span_exporter is not None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if instrumented:
            FastAPIInstrumentor.uninstrument_app(app)
        shutdown_tracing(tracer_provider)
        shutdown_meter_provider(meter_provider)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.tracer_provider = tracer_provider
    app.state.meter_provider = meter_provider
    app.state.meter = meter
    app.state.app_metrics = app_metrics

    health = health_service or build_health_service(settings)
    tracer = tracer_provider.get_tracer(settings.otel_service_name)
    health.tracer = tracer
    app.state.health_service = health

    app.include_router(health_router)

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{settings.app_name} API",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc() -> HTMLResponse:
        return get_redoc_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{settings.app_name} API",
        )

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        request_id = getattr(request.state, "groundgraph_request_id", None)
        if not isinstance(request_id, str):
            request_id = request_id_from_headers(list(request.scope.get("headers", [])))
        correlation_id = request_id
        stored_span = getattr(request.state, "groundgraph_server_span", None)
        span = stored_span if isinstance(stored_span, Span) else trace.get_current_span()
        span_context = span.get_span_context()
        bind_request_context(
            {
                "request_id": request_id,
                "correlation_id": correlation_id,
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
            }
        )
        status_code: int | None = None
        response: Response | None = None
        error_type: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = SERVER_ERROR_STATUS
            error_type = type(exc).__name__
        finally:
            duration = time.perf_counter() - started
            route = _safe_route(request)
            for key, value in sanitize_attributes(
                {
                    "groundgraph.request_id": request_id,
                    "groundgraph.correlation_id": correlation_id,
                    "http.route": route,
                    "http.response.status_code": status_code,
                }
            ).items():
                span.set_attribute(key, value)

            if not _is_excluded_route(route):
                metrics = app_metrics
                metrics.http_request_count.add(
                    1, {"method": request.method, "route": route, "status_code": str(status_code)}
                )
                metrics.http_request_duration.record(
                    duration, {"method": request.method, "route": route}
                )
                if status_code is not None and status_code >= SERVER_ERROR_STATUS:
                    metrics.http_request_errors.add(
                        1,
                        {
                            "method": request.method,
                            "route": route,
                            "exception_type": error_type or "server_error",
                        },
                    )
                _log_request(
                    "http.request.completed" if error_type is None else "http.request.failed",
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    duration_seconds=duration,
                    error_type=error_type,
                )

            clear_request_context()

        if response is None:
            response = JSONResponse(
                status_code=SERVER_ERROR_STATUS,
                content={"code": "internal_error", "message": "Internal server error"},
            )
            status_code = SERVER_ERROR_STATUS

        response.headers["x-request-id"] = request_id
        response.headers["x-correlation-id"] = correlation_id
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
        return response

    if instrumented:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer_provider,
            excluded_urls="health/live",
            http_capture_headers_server_request=[],
            http_capture_headers_server_response=[],
            http_capture_headers_sanitize_fields=["authorization", "cookie", "set-cookie"],
            exclude_spans=["receive", "send"],
            server_request_hook=_server_request_hook,
        )

    return app
