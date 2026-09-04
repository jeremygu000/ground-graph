"""Tests for the production fail-closed configuration validator.

The plan and the security reviewer require that the application refuses
to start in production with obviously misconfigured secrets. These
tests assert that the validator catches each documented misconfiguration.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from groundgraph.application.settings import Settings


def _base_production_kwargs() -> dict[str, Any]:
    return {
        "app_env": "production",
        "openai_api_key": SecretStr("sk-real-key-1234567890"),
        "postgres_password": SecretStr("real-pg-password"),
        "neo4j_password": SecretStr("real-neo4j-password"),
        "s3_secret_key": SecretStr("real-s3-secret"),
        "otel_exporter_otlp_insecure": False,
        "auth_mode": "oidc",
        "auth_trusted_headers": False,
    }


def test_production_with_valid_secrets_succeeds() -> None:
    settings = Settings(**_base_production_kwargs())
    assert settings.is_production is True


def test_production_rejects_empty_openai_key() -> None:
    kwargs = _base_production_kwargs()
    kwargs["openai_api_key"] = SecretStr("")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(**kwargs)


def test_production_rejects_placeholder_postgres_password() -> None:
    kwargs = _base_production_kwargs()
    kwargs["postgres_password"] = SecretStr("change-me-local-only")
    with pytest.raises(ValueError, match="placeholder secret"):
        Settings(**kwargs)


def test_production_rejects_placeholder_neo4j_password() -> None:
    kwargs = _base_production_kwargs()
    kwargs["neo4j_password"] = SecretStr("change-me-local-only")
    with pytest.raises(ValueError, match="placeholder secret"):
        Settings(**kwargs)


def test_production_rejects_placeholder_s3_secret() -> None:
    kwargs = _base_production_kwargs()
    kwargs["s3_secret_key"] = SecretStr("change-me-local-only")
    with pytest.raises(ValueError, match="placeholder secret"):
        Settings(**kwargs)


def test_production_rejects_insecure_otlp() -> None:
    kwargs = _base_production_kwargs()
    kwargs["otel_exporter_otlp_insecure"] = True
    with pytest.raises(ValueError, match="INSECURE"):
        Settings(**kwargs)


def test_production_rejects_local_auth_mode() -> None:
    kwargs = _base_production_kwargs()
    kwargs["auth_mode"] = "local"
    with pytest.raises(ValueError, match="AUTH_MODE"):
        Settings(**kwargs)


def test_production_rejects_trusted_headers() -> None:
    kwargs = _base_production_kwargs()
    kwargs["auth_trusted_headers"] = True
    with pytest.raises(ValueError, match="AUTH_TRUSTED_HEADERS"):
        Settings(**kwargs)


def test_development_allows_default_secrets() -> None:
    """The default development config must remain easy to run locally."""
    settings = Settings()
    assert settings.is_production is False
    assert settings.openai_api_key_value == ""


def test_test_environment_allows_default_secrets() -> None:
    settings = Settings(app_env="test")
    assert settings.is_test is True


def test_production_reports_all_problems_at_once() -> None:
    """A single error message should mention every failure category."""
    kwargs: dict[str, Any] = {
        "app_env": "production",
        "openai_api_key": SecretStr(""),
        "postgres_password": SecretStr("change-me-local-only"),
        "otel_exporter_otlp_insecure": True,
        "auth_mode": "local",
    }
    with pytest.raises(ValueError, match="Production configuration is unsafe") as exc:
        Settings(**kwargs)
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert "placeholder secret" in msg
    assert "INSECURE" in msg
    assert "AUTH_MODE" in msg
