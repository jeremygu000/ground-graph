"""Telemetry configuration and safe redaction utilities."""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

from opentelemetry import context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Tracer

SENSITIVE_ATTRIBUTE_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "proxy-authorization",
        "request.body",
        "response.body",
        "http.request.body",
        "http.response.body",
    }
)
SENSITIVE_ATTRIBUTE_TERMS = (
    "authorization",
    "api-key",
    "cookie",
    "secret",
    "token",
    "password",
)


def configure_tracing(
    service_name: str,
    otlp_endpoint: str | None,
    *,
    exporter: SpanExporter | None = None,
    enable_otlp: bool = True,
    otlp_insecure: bool = False,
) -> TracerProvider:
    """Create an app-local tracer provider with optional OTLP export.

    The provider is deliberately not installed globally. App factories are
    invoked repeatedly by tests, and FastAPI instrumentation accepts this
    provider directly without global state or background exporter leaks.
    """

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if enable_otlp or exporter is not None:
        span_exporter = exporter
        if span_exporter is None:
            if otlp_endpoint is None:
                raise ValueError("otlp_endpoint is required when OTLP is enabled")
            span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=otlp_insecure)
        processor = (
            SimpleSpanProcessor(span_exporter)
            if exporter is not None
            else BatchSpanProcessor(span_exporter)
        )
        provider.add_span_processor(processor)
    return provider


def traced_background_task(
    tracer: Tracer,
    name: str,
    callback: Any,
) -> Any:
    """Wrap a background callback so its span retains the request context."""

    parent_context = context.get_current()

    async def run() -> None:
        token = context.attach(parent_context)
        try:
            with tracer.start_as_current_span(name):
                result = callback()
                if inspect.isawaitable(result):
                    await result
        finally:
            context.detach(token)

    return run


def redact_text(value: str, patterns: Iterable[str]) -> str:
    """Remove obvious secret-like content from text before it reaches telemetry."""

    redacted = value
    for pattern in patterns:
        if pattern and pattern.lower() in redacted.lower():
            redacted = re.sub(re.escape(pattern), "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Return a telemetry-safe attribute subset.

    Bodies and credential-bearing HTTP headers are omitted by default;
    callers may only pass explicit scalar metadata to this helper.
    """
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized = key.lower().replace("_", "-")
        if (
            normalized in SENSITIVE_ATTRIBUTE_NAMES
            or any(term in normalized for term in SENSITIVE_ATTRIBUTE_TERMS)
            or normalized.endswith("body")
        ):
            continue
        if isinstance(value, (str, bool, int, float)):
            safe[key] = value
    return safe


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Best-effort shutdown for tests."""

    if provider is None:
        return
    with suppress(Exception):
        provider.shutdown()
