"""Tests for the application settings module."""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from groundgraph.application.settings import Settings, get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a clean settings load by clearing the lru_cache per-test."""
    for key in list(os.environ):
        if key.startswith(
            (
                "APP_",
                "API_",
                "POSTGRES_",
                "NEO4J_",
                "S3_",
                "OPENAI_",
                "OTEL_",
                "PHOENIX_",
                "PROMETHEUS_",
                "VECTOR_",
                "KEYWORD_",
                "GRAPH_",
                "FINAL_",
                "RETRIEVAL_",
                "TELEMETRY_",
                "AUTH_",
                "ONTOLOGY_",
                "INDEX_",
                "PROMPT_",
                "LOG_",
                "EMBEDDING_",
                "GENERATION_",
                "EXTRACTION_",
                "PLANNER_",
                "JUDGE_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()


def test_settings_loads_with_defaults() -> None:
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.graph_depth_max <= 3
    assert settings.postgres_password.get_secret_value() == "change-me-local-only"


def test_settings_postgres_dsn_async() -> None:
    settings = Settings(
        postgres_host="db",
        postgres_port=5433,
        postgres_db="rag",
        postgres_user="u",
        postgres_password=SecretStr("pw"),
    )
    assert settings.postgres_dsn_async == "postgresql+asyncpg://u:pw@db:5433/rag"


def test_settings_postgres_dsn_sync() -> None:
    settings = Settings(
        postgres_host="db",
        postgres_port=5433,
        postgres_db="rag",
        postgres_user="u",
        postgres_password=SecretStr("pw"),
    )
    assert settings.postgres_dsn_sync == "postgresql+psycopg://u:pw@db:5433/rag"


def test_settings_rejects_graph_depth_above_three() -> None:
    with pytest.raises(ValueError, match="graph_depth_max"):
        Settings(graph_depth_max=5)


def test_settings_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="telemetry_sample_rate"):
        Settings(telemetry_sample_rate=1.5)


def test_settings_cors_origins_list_parses_csv() -> None:
    settings = Settings(api_cors_origins="http://a, http://b ,http://c")
    assert settings.cors_origins_list == ["http://a", "http://b", "http://c"]


def test_settings_redact_patterns_list_parses_csv() -> None:
    settings = Settings(telemetry_redact_patterns="password, token ,api_key")
    assert settings.redact_patterns_list == ["password", "token", "api_key"]


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings()
    reset_settings_cache()
    monkeypatch.setenv("APP_ENV", "test")
    s = get_settings()
    assert s.app_env == "test"


def test_settings_typed_properties() -> None:
    settings = Settings()
    assert isinstance(settings.is_production, bool)
    assert isinstance(settings.is_test, bool)
    assert settings.is_production is False
    assert settings.is_test is False


def test_settings_openai_key_value_empty_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[reportCallIssue]
    assert settings.openai_api_key_value == ""
