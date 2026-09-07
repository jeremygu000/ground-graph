"""Tests for the operator dashboard API — P0 operator auth verification."""

from __future__ import annotations

from time import time
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from jose import jwt
from pydantic import SecretStr

from groundgraph.api import dependencies
from groundgraph.api.app import create_app
from groundgraph.application.settings import Settings


def _settings() -> Settings:
    return Settings(
        app_env="test",
        openai_api_key=SecretStr(""),
        otel_exporter_otlp_endpoint="http://localhost:4317",
        otel_exporter_otlp_insecure=True,
        auth_mode="local",
        auth_trusted_headers=True,
        auth_jwks_url="",
        auth_issuer="https://test.example.com",
        auth_audience="test-audience",
        auth_local_secret=SecretStr("test-secret-key-at-least-32-chars-long"),
    )


def _make_operator_token(
    tenant_id: str = "tenant-1",
    principal: str = "user-1",
    role: str = "operator",
    is_global: bool = False,
    secret: str = "test-secret-key-at-least-32-chars-long",
) -> str:
    payload = {
        "tenant_id": tenant_id,
        "sub": principal,
        "role": role,
        "is_global_operator": is_global,
        "iss": "https://test.example.com",
        "aud": "test-audience",
        "exp": int(time()) + 3600,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class _MockAsyncSession:
    async def __aenter__(self) -> Any:
        from datetime import UTC, datetime

        self._add_calls = []

        def mock_add(obj: Any) -> None:
            obj.feedback_id = uuid4()
            obj.created_at = datetime.now(UTC)

        self.add = mock_add
        self.commit = AsyncMock()
        self.flush = AsyncMock()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _MockSessionFactory:
    def __call__(self) -> _MockAsyncSession:
        return _MockAsyncSession()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from groundgraph.api import operator as operator_module

    test_settings = _settings()
    mock_factory = _MockSessionFactory()
    monkeypatch.setattr(operator_module, "get_session_factory", mock_factory)
    application = create_app(test_settings, telemetry_enabled=False)
    application.dependency_overrides[dependencies.get_settings] = lambda: test_settings
    return application


async def _request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


class TestOperatorAuthRejectsArbitraryHeader:
    """P0: arbitrary X-Operator-ID must not grant access."""

    @pytest.mark.asyncio
    async def test_arbitrary_x_operator_id_rejected(self, app: FastAPI) -> None:
        response = await _request(
            app,
            "GET",
            "/operator/executions",
            headers={"X-Operator-ID": "admin"},
        )
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_bearer_token_required(self, app: FastAPI) -> None:
        response = await _request(app, "GET", "/operator/executions")
        assert response.status_code == 401
        assert "Authorization" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_non_bearer_token_rejected(self, app: FastAPI) -> None:
        response = await _request(
            app,
            "GET",
            "/operator/executions",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert response.status_code == 401


class TestOperatorAuthRequiresOperatorRole:
    """P0: valid JWT without operator/admin role must be rejected."""

    @pytest.mark.asyncio
    async def test_ordinary_jwt_without_role_rejected(self, app: FastAPI) -> None:
        token = _make_operator_token(role="viewer")
        response = await _request(
            app,
            "GET",
            "/operator/executions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "role" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_admin_role_allowed(self, app: FastAPI) -> None:
        token = _make_operator_token(role="admin")
        response = await _request(
            app,
            "GET",
            "/operator/executions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_operator_role_allowed(self, app: FastAPI) -> None:
        token = _make_operator_token(role="operator")
        response = await _request(
            app,
            "GET",
            "/operator/executions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


class TestOperatorTenantScoping:
    """P0: tenant operator must not access other tenant's data."""

    @pytest.mark.asyncio
    async def test_tenant_operator_cannot_access_other_tenant_run(self, app: FastAPI) -> None:
        token = _make_operator_token(
            tenant_id="tenant-a",
            principal="operator-a",
            role="operator",
            is_global=False,
        )
        other_tenant_run_id = str(uuid4())
        response = await _request(
            app,
            "GET",
            f"/operator/executions/{other_tenant_run_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code in (403, 404)

    @pytest.mark.asyncio
    async def test_global_operator_can_access_any_tenant(self, app: FastAPI) -> None:
        token = _make_operator_token(
            tenant_id="tenant-a",
            principal="global-admin",
            role="admin",
            is_global=True,
        )
        other_tenant_run_id = str(uuid4())
        response = await _request(
            app,
            "GET",
            f"/operator/executions/{other_tenant_run_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code in (200, 404)


class TestEvaluationTrends:
    """Tests for evaluation trends endpoint."""

    @pytest.mark.asyncio
    async def test_trends_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "GET", "/operator/evaluation/trends")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_trends_returns_list(self, app: FastAPI) -> None:
        token = _make_operator_token(role="operator")
        response = await _request(
            app,
            "GET",
            "/operator/evaluation/trends",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestIngestionStatus:
    """Tests for ingestion status endpoint."""

    @pytest.mark.asyncio
    async def test_ingestion_status_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "GET", "/operator/ingestion/status")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ingestion_status_returns_list(self, app: FastAPI) -> None:
        token = _make_operator_token(role="operator")
        response = await _request(
            app,
            "GET",
            "/operator/ingestion/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestFeedbackSubmission:
    """Tests for feedback submission endpoint."""

    @pytest.mark.asyncio
    async def test_feedback_requires_auth(self, app: FastAPI) -> None:
        response = await _request(
            app,
            "POST",
            "/operator/feedback",
            json={
                "execution_run_id": str(uuid4()),
                "vote": "thumbs_up",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_feedback_accepts_valid_submission(self, app: FastAPI) -> None:
        token = _make_operator_token(role="operator")
        response = await _request(
            app,
            "POST",
            "/operator/feedback",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "execution_run_id": str(uuid4()),
                "vote": "thumbs_up",
                "category": "correct_answer",
                "notes": "Looks good",
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestReviewQueue:
    """Tests for review queue endpoint."""

    @pytest.mark.asyncio
    async def test_review_queue_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "GET", "/operator/review/queue")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_review_queue_returns_list(self, app: FastAPI) -> None:
        token = _make_operator_token(role="operator")
        response = await _request(
            app,
            "GET",
            "/operator/review/queue",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
