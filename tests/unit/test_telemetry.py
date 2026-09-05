"""Unit tests for telemetry helpers.

These tests focus on the redaction/sanitization utilities and the
provider-construction paths that don't require a live OTLP collector.
The OTLP exporter constructor is exercised through a fake injection.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from groundgraph.infrastructure.telemetry import (
    SENSITIVE_ATTRIBUTE_NAMES,
    configure_meter_provider,
    configure_tracing,
    get_meter,
    redact_text,
    sanitize_attributes,
    shutdown_meter_provider,
    shutdown_tracing,
    traced_background_task,
)


class _InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.spans.clear()


def test_configure_tracing_with_injected_exporter() -> None:
    exporter = _InMemorySpanExporter()
    provider = configure_tracing("svc", None, exporter=exporter)
    assert isinstance(provider, TracerProvider)
    shutdown_tracing(provider)


def test_configure_tracing_without_exporter_returns_provider() -> None:
    provider = configure_tracing("svc", None, enable_otlp=False)
    assert isinstance(provider, TracerProvider)
    shutdown_tracing(provider)


def test_configure_tracing_otlp_endpoint_required_when_otlp_enabled() -> None:
    with pytest.raises(ValueError, match="otlp_endpoint is required"):
        configure_tracing("svc", None, enable_otlp=True, exporter=None)


def test_configure_tracing_otlp_endpoint_constructs_exporter() -> None:
    provider = configure_tracing("svc", "http://collector:4317", enable_otlp=True)
    assert isinstance(provider, TracerProvider)
    shutdown_tracing(provider)


def test_shutdown_tracing_handles_none() -> None:
    shutdown_tracing(None)


def test_shutdown_tracing_swallows_exceptions() -> None:
    class _Boom:
        def shutdown(self) -> None:
            raise RuntimeError("boom")

    shutdown_tracing(cast(Any, _Boom()))


@pytest.mark.asyncio
async def test_traced_background_task_runs_sync_callback() -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    captured: list[str] = []

    def cb() -> str:
        captured.append("ran")
        return "ok"

    runner = traced_background_task(tracer, "sync-op", cb)
    assert asyncio.iscoroutinefunction(runner)
    await runner()
    assert captured == ["ran"]


@pytest.mark.asyncio
async def test_traced_background_task_runs_async_callback() -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    captured: list[str] = []

    async def acb() -> None:
        captured.append("async")

    runner = traced_background_task(tracer, "async-op", acb)
    await runner()
    assert captured == ["async"]


def test_redact_text_substitutes_known_patterns() -> None:
    text = "user=Bearer abcdef token=xyz123; payload=ok"
    out = redact_text(text, ["Bearer abcdef"])
    assert "Bearer abcdef" not in out
    assert "[REDACTED]" in out
    assert "payload=ok" in out


def test_redact_text_handles_empty_pattern_and_no_match() -> None:
    assert redact_text("safe text", []) == "safe text"
    assert redact_text("nothing here", ["missing"]) == "nothing here"


def test_redact_text_skips_empty_or_whitespace_pattern() -> None:
    out = redact_text("hello world", ["", "  "])
    assert out == "hello world"


def test_sanitize_attributes_drops_sensitive_keys() -> None:
    attrs = {
        "authorization": "Bearer xyz",
        "api_key": "k",
        "cookie": "c",
        "x_token": "t",
        "request_body": "leak",
        "user_id": 7,
        "trace": "ok",
    }
    safe = sanitize_attributes(attrs)
    assert "authorization" not in safe
    assert "api_key" not in safe
    assert "cookie" not in safe
    assert "x_token" not in safe
    assert "request_body" not in safe
    assert "user_id" in safe
    assert "trace" in safe


def test_sanitize_attributes_drops_non_scalar_values() -> None:
    attrs: dict[str, Any] = {"name": "ok", "blob": b"raw", "items": [1, 2, 3], "obj": {"k": "v"}}
    safe = sanitize_attributes(attrs)
    assert safe == {"name": "ok"}


def test_sanitize_attributes_treats_dashes_underscores_equivalently() -> None:
    attrs = {"request-body": "leak", "request_body": "leak", "Request_Body": "leak"}
    safe = sanitize_attributes(attrs)
    assert safe == {}


def test_sanitize_attributes_normalises_key() -> None:
    attrs = {"x_token": "t", "X-Token": "t", "X_TOKEN": "t"}
    safe = sanitize_attributes(attrs)
    assert safe == {}


def test_sanitize_attributes_drops_body_suffix_keys() -> None:
    attrs = {"foo_body": "leak", "Foo-Body": "leak", "request_body": "leak"}
    safe = sanitize_attributes(attrs)
    assert safe == {}


def test_sensitive_attribute_names_is_frozenset() -> None:
    assert isinstance(SENSITIVE_ATTRIBUTE_NAMES, frozenset)
    assert "authorization" in SENSITIVE_ATTRIBUTE_NAMES


def test_configure_meter_provider_with_reader() -> None:
    reader = InMemoryMetricReader()
    provider = configure_meter_provider("svc", None, metric_reader=reader)
    assert isinstance(provider, MeterProvider)
    shutdown_meter_provider(provider)


def test_configure_meter_provider_otlp_branch() -> None:
    provider = configure_meter_provider("svc", "http://collector:4317", enable_otlp=True)
    assert isinstance(provider, MeterProvider)
    shutdown_meter_provider(provider)


def test_configure_meter_provider_default_branch() -> None:
    provider = configure_meter_provider("svc", None, enable_otlp=False)
    assert isinstance(provider, MeterProvider)
    shutdown_meter_provider(provider)


def test_get_meter_returns_meter() -> None:
    provider = configure_meter_provider("svc", None, enable_otlp=False)
    try:
        meter = get_meter(provider, "test")
        assert meter is not None
    finally:
        shutdown_meter_provider(provider)


def test_shutdown_meter_provider_handles_none() -> None:
    shutdown_meter_provider(None)


def test_shutdown_meter_provider_swallows_exceptions() -> None:
    class _Boom:
        def shutdown(self) -> None:
            raise RuntimeError("boom")

    shutdown_meter_provider(cast(Any, _Boom()))
