"""Tests for the operator dashboard API."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from groundgraph.api.app import create_app
from groundgraph.application.settings import Settings


def _settings() -> Settings:
    return Settings(
        app_env="test",
        openai_api_key=SecretStr(""),
        otel_exporter_otlp_endpoint="http://localhost:4317",
        otel_exporter_otlp_insecure=True,
        auth_mode="local",
        auth_trusted_headers=False,
    )


@pytest.fixture
def app() -> FastAPI:
    return create_app(_settings(), telemetry_enabled=False)


async def _request(app: FastAPI, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request("GET", path, **kwargs)


async def _post(app: FastAPI, path: str, **kwargs: Any) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request("POST", path, **kwargs)


class TestOperatorAuth:
    """Tests for operator authentication."""

    @pytest.mark.asyncio
    async def test_missing_operator_header_returns_401(self, app: FastAPI) -> None:
        response = await _request(app, "/operator/executions")
        assert response.status_code == 401
        assert "X-Operator-ID" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_empty_operator_header_returns_401(self, app: FastAPI) -> None:
        response = await _request(app, "/operator/executions", headers={"X-Operator-ID": ""})
        assert response.status_code == 401


class TestEvaluationTrends:
    """Tests for evaluation trends endpoint."""

    @pytest.mark.asyncio
    async def test_trends_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "/operator/evaluation/trends")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_trends_returns_list(self, app: FastAPI) -> None:
        response = await _request(
            app, "/operator/evaluation/trends", headers={"X-Operator-ID": "operator-1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "metric" in data[0]
            assert "current_score" in data[0]


class TestIngestionStatus:
    """Tests for ingestion status endpoint."""

    @pytest.mark.asyncio
    async def test_ingestion_status_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "/operator/ingestion/status")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_ingestion_status_returns_list(self, app: FastAPI) -> None:
        response = await _request(
            app, "/operator/ingestion/status", headers={"X-Operator-ID": "operator-1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestFeedbackSubmission:
    """Tests for feedback submission endpoint."""

    @pytest.mark.asyncio
    async def test_feedback_requires_auth(self, app: FastAPI) -> None:
        response = await _post(
            app,
            "/operator/feedback",
            json={
                "execution_run_id": str(uuid4()),
                "category": "correct_answer",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_feedback_accepts_valid_submission(self, app: FastAPI) -> None:
        response = await _post(
            app,
            "/operator/feedback",
            headers={"X-Operator-ID": "operator-1"},
            json={
                "execution_run_id": str(uuid4()),
                "category": "correct_answer",
                "notes": "Looks good",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "feedback_id" in data
        assert data["status"] == "recorded"


class TestReviewQueue:
    """Tests for review queue endpoint."""

    @pytest.mark.asyncio
    async def test_review_queue_requires_auth(self, app: FastAPI) -> None:
        response = await _request(app, "/operator/review/queue")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_review_queue_returns_list(self, app: FastAPI) -> None:
        response = await _request(
            app, "/operator/review/queue", headers={"X-Operator-ID": "operator-1"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
