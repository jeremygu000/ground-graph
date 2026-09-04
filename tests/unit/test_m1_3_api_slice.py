"""M1.3 API and telemetry slice tests."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from groundgraph.api.app import create_app
from groundgraph.application.settings import Settings
from groundgraph.infrastructure.telemetry import redact_text


def _settings() -> Settings:
    return Settings(
        app_env="test",
        openai_api_key=SecretStr(""),
        otel_exporter_otlp_endpoint="http://localhost:4317",
        otel_exporter_otlp_insecure=True,
        auth_mode="local",
        auth_trusted_headers=False,
    )


@pytest.mark.anyio
async def test_create_app_exposes_health_routes() -> None:
    app = create_app(_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 200


@pytest.mark.anyio
async def test_create_app_injects_request_ids() -> None:
    app = create_app(_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/live", headers={"x-request-id": "req-123"})
        assert response.headers["x-request-id"] == "req-123"
        assert response.headers["x-correlation-id"] == "req-123"
        assert response.headers["x-process-time-ms"]


def test_redact_text_replaces_secret_like_terms() -> None:
    assert redact_text("Authorization token secret", ["authorization", "token", "secret"]) == (
        "[REDACTED] [REDACTED] [REDACTED]"
    )
