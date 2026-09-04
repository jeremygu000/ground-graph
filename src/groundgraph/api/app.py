"""FastAPI composition root for the M1.3 observability slice."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from groundgraph.api.dependencies import build_health_service, request_id_from_headers
from groundgraph.api.health import router as health_router
from groundgraph.application.health import HealthService
from groundgraph.application.settings import Settings, get_settings
from groundgraph.infrastructure.logging import (
    bind_request_context,
    clear_request_context,
    configure_json_logging,
)
from groundgraph.infrastructure.metrics import (
    HTTP_REQUEST_COUNT,
    HTTP_REQUEST_DURATION,
    HTTP_REQUEST_ERRORS,
)
from groundgraph.infrastructure.telemetry import (
    configure_tracing,
    sanitize_attributes,
    shutdown_tracing,
)

SERVER_ERROR_STATUS = 500


def _safe_route(request: Request) -> str:
    route = getattr(request.scope.get("route"), "path", None)
    return route or request.url.path


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
    telemetry_enabled: bool | None = None,
) -> FastAPI:
    """Create the API application.

    The app is test-friendly: tracing can be disabled for unit tests,
    and the process-wide tracer provider is only set once.
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

    instrumented = tracing_enabled or span_exporter is not None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if instrumented:
            FastAPIInstrumentor.uninstrument_app(app)
        shutdown_tracing(tracer_provider)

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.tracer_provider = tracer_provider
    app.state.health_service = health_service or build_health_service(settings)
    app.include_router(health_router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Expose Prometheus metrics without accepting request content."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
        try:
            response = await call_next(request)
            route = _safe_route(request)
            duration = time.perf_counter() - started
            response.headers["x-request-id"] = request_id
            response.headers["x-correlation-id"] = correlation_id
            response.headers["x-process-time-ms"] = f"{duration * 1000:.3f}"
            for key, value in sanitize_attributes(
                {
                    "groundgraph.request_id": request_id,
                    "groundgraph.correlation_id": correlation_id,
                    "http.route": route,
                    "http.response.status_code": response.status_code,
                }
            ).items():
                span.set_attribute(key, value)
            HTTP_REQUEST_COUNT.labels(request.method, route, str(response.status_code)).inc()
            HTTP_REQUEST_DURATION.labels(request.method, route).observe(duration)
            if response.status_code >= SERVER_ERROR_STATUS:
                HTTP_REQUEST_ERRORS.labels(request.method, route, "server_error").inc()
                _log_request(
                    "http.request.failed",
                    method=request.method,
                    route=route,
                    status_code=response.status_code,
                    duration_seconds=duration,
                )
            else:
                _log_request(
                    "http.request.completed",
                    method=request.method,
                    route=route,
                    status_code=response.status_code,
                    duration_seconds=duration,
                )
        except Exception:
            route = _safe_route(request)
            HTTP_REQUEST_ERRORS.labels(request.method, route, "internal_error").inc()
            _log_request(
                "http.request.failed",
                method=request.method,
                route=route,
                error_type="internal_error",
            )
            raise
        finally:
            clear_request_context()
        return response

    @app.exception_handler(Exception)
    async def _exception_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=SERVER_ERROR_STATUS,
            content={"code": "internal_error", "message": "Internal server error"},
        )

    # Add the OpenTelemetry middleware last, making it the outer request
    # middleware. The request context middleware can then enrich its server span.
    if instrumented:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer_provider,
            excluded_urls="health/live,health/ready",
            http_capture_headers_server_request=[],
            http_capture_headers_server_response=[],
            http_capture_headers_sanitize_fields=["authorization", "cookie", "set-cookie"],
            exclude_spans=["receive", "send"],
            server_request_hook=_server_request_hook,
        )

    return app
