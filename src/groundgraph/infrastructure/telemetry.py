"""Telemetry configuration and safe redaction utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing(service_name: str, otlp_endpoint: str) -> None:
    """Install a process-wide tracer provider exporting to OTLP."""

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def redact_text(value: str, patterns: Iterable[str]) -> str:
    """Remove obvious secret-like content from text before it reaches telemetry."""

    redacted = value
    for pattern in patterns:
        if pattern and pattern.lower() in redacted.lower():
            redacted = re.sub(re.escape(pattern), "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted
