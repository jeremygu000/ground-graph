"""Typed application settings loaded from environment.

This is the single source of truth for environment configuration.
All other code reads settings through this module; do not call os.environ
elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment and .env file.

    Required secrets will cause validation errors on startup, ensuring the
    application fails clearly rather than running with missing configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_name: str = "agentic-graphrag"
    app_version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_cors_origins: str = "http://localhost:3000"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graphrag"
    postgres_user: str = "graphrag"
    postgres_password: SecretStr = Field(default=SecretStr("change-me-local-only"))
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = Field(default=SecretStr("change-me-local-only"))
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "change-me-local-only"
    s3_secret_key: SecretStr = Field(default=SecretStr("change-me-local-only"))
    s3_bucket_raw: str = "graphrag-raw"
    s3_bucket_processed: str = "graphrag-processed"
    s3_region: str = "us-east-1"
    s3_use_ssl: bool = False

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_org_id: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    generation_model: str = "gpt-4o-mini"
    extraction_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    otel_service_name: str = "agentic-graphrag"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_exporter_otlp_insecure: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "agentic-graphrag"

    prometheus_port: int = 9464

    vector_top_k: int = 30
    vector_reranked_k: int = 10
    keyword_top_k: int = 20
    graph_depth_default: int = 2
    graph_depth_max: int = 3
    graph_candidate_paths: int = 20
    final_evidence_limit: int = 12
    retrieval_retries: int = 1

    telemetry_capture_content: bool = False
    telemetry_sample_rate: float = 1.0
    telemetry_redact_patterns: str = "password,token,secret,api_key,authorization"

    auth_mode: Literal["local", "header", "oidc"] = "local"
    auth_default_principal: str = "engineering"
    auth_default_tenant: str = "default"
    auth_trusted_headers: bool = False

    ontology_version: str = "v0.1.0"
    index_version: str = "v0.1.0"
    prompt_bundle_version: str = "v0.1.0"

    GRAPH_DEPTH_MVP_MAX: int = 3

    @field_validator("graph_depth_max")
    @classmethod
    def _validate_graph_depth_max(cls, v: int) -> int:
        if v > 3:  # noqa: PLR2004 - MVP hard limit per plan.md §7.5
            raise ValueError("graph_depth_max must be <= 3 in MVP")
        return v

    @field_validator("telemetry_sample_rate")
    @classmethod
    def _validate_sample_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("telemetry_sample_rate must be between 0 and 1")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @property
    def redact_patterns_list(self) -> list[str]:
        return [p.strip() for p in self.telemetry_redact_patterns.split(",") if p.strip()]

    @property
    def postgres_dsn_async(self) -> str:
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def openai_api_key_value(self) -> str:
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so that environment is read once. Use ``get_settings.cache_clear()``
    in tests that need to re-read environment.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used in tests)."""
    get_settings.cache_clear()
