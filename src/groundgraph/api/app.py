"""FastAPI application composition root."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from groundgraph.api.health import router as health_router
from groundgraph.application.settings import Settings, get_settings
from groundgraph.infrastructure.logging import configure_json_logging
from groundgraph.infrastructure.telemetry import configure_tracing, redact_text


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_json_logging(settings.log_level)
    configure_tracing(settings.otel_service_name, settings.otel_exporter_otlp_endpoint)

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=None)
    app.include_router(health_router)

    @app.middleware("http")
    async def request_context_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        correlation_id = request.headers.get("x-correlation-id") or request_id
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("http.request", kind=SpanKind.SERVER) as span:
            span.set_attribute("http.request_id", request_id)
            span.set_attribute("http.correlation_id", correlation_id)
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)

            start = time.perf_counter()
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000.0
            response.headers["x-request-id"] = request_id
            response.headers["x-correlation-id"] = correlation_id
            response.headers["x-process-time-ms"] = f"{duration_ms:.2f}"
            response.headers.setdefault("content-type", "application/json")
            response.headers["x-telemetry-redaction-policy"] = ",".join(
                settings.redact_patterns_list
            )
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("http.duration_ms", duration_ms)
            span.set_attribute(
                "http.route.safe", redact_text(request.url.path, settings.redact_patterns_list)
            )
            return response

    @app.exception_handler(Exception)
    async def _fallback_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Avoid leaking raw exception details to the caller.
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "Internal server error"},
        )

    return app
