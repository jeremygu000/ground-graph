"""Local infrastructure validation tests.

These tests verify the docker-compose configuration is well-formed and
that the listed services match the dependency requirements in plan.md §M1.
The tests do NOT require Docker to run; they only need the file to be valid YAML.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.yml"


REQUIRED_SERVICES = {
    "postgres": "PostgreSQL + pgvector (plan.md §M1)",
    "neo4j": "Neo4j (plan.md §M1)",
    "minio": "MinIO S3-compatible storage (plan.md §M1)",
    "phoenix": "Arize Phoenix AI trace UI (plan.md §M1)",
    "otel-collector": "OpenTelemetry Collector (plan.md §M1)",
    "prometheus": "Prometheus (plan.md §M1)",
    "grafana": "Grafana (plan.md §M1)",
}


def _yaml_load(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


def test_docker_compose_file_exists() -> None:
    assert COMPOSE_FILE.exists(), f"docker-compose.yml not found at {COMPOSE_FILE}"


def test_docker_compose_is_valid_yaml() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    assert isinstance(parsed, dict)
    assert "services" in parsed
    assert "networks" in parsed
    assert "volumes" in parsed


def test_docker_compose_required_services_present() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    services = parsed.get("services", {})
    missing = sorted(name for name in REQUIRED_SERVICES if name not in services)
    assert not missing, f"Required services missing from docker-compose: {missing}"


def test_docker_compose_services_have_healthchecks() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    services = parsed.get("services", {})
    for name, svc in services.items():
        if name.endswith("-init") or name in {"minio-init"}:
            # one-shot init jobs legitimately lack healthchecks
            continue
        assert "healthcheck" in svc, f"Service {name} has no healthcheck"


def test_docker_compose_uses_named_volumes_for_state() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    volumes = parsed.get("volumes", {})
    expected = {
        "postgres-data",
        "neo4j-data",
        "minio-data",
        "phoenix-data",
        "prometheus-data",
        "grafana-data",
    }
    missing = sorted(expected - set(volumes.keys()))
    assert not missing, f"Missing named volumes: {missing}"


def test_docker_compose_exposes_standard_ports() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    services = parsed.get("services", {})

    def _published(svc: dict) -> set[str]:
        out: set[str] = set()
        for port in svc.get("ports", []) or []:
            if isinstance(port, str):
                out.add(port.split(":")[-1].split("/")[0])
            elif isinstance(port, dict):
                out.add(str(port.get("published")))
        return out

    # Sanity: each major service exposes its canonical port.
    assert "5432" in _published(services["postgres"]), "Postgres must expose 5432"
    assert "7687" in _published(services["neo4j"]), "Neo4j must expose 7687 (bolt)"
    assert "9000" in _published(services["minio"]), "MinIO must expose 9000 (S3)"
    assert "6006" in _published(services["phoenix"]), "Phoenix must expose 6006"
    assert "4317" in _published(services["otel-collector"]), (
        "OTel collector must expose 4317 (OTLP gRPC)"
    )
    assert "9090" in _published(services["prometheus"]), "Prometheus must expose 9090"
    assert "3000" in _published(services["grafana"]), "Grafana must expose 3000"


@pytest.mark.skipif(not shutil.which("docker"), reason="docker CLI not available")
def test_docker_compose_config_passes() -> None:
    """If docker is available, run `docker compose config` to validate fully."""
    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and "cannot find docker" in (proc.stderr or "").lower():
        pytest.skip("docker daemon not reachable")
    assert proc.returncode == 0, f"docker compose config failed: {proc.stdout}\n{proc.stderr}"


def test_env_example_has_required_keys() -> None:
    env_file = ROOT / ".env.example"
    assert env_file.exists()
    text = env_file.read_text(encoding="utf-8")
    required_keys = {
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "PHOENIX_COLLECTOR_ENDPOINT",
    }
    for key in required_keys:
        assert key in text, f".env.example missing required key {key}"


def test_env_example_does_not_commit_real_secrets() -> None:
    """Defensive check: the example must not contain real-looking secrets."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    # Placeholder values like "change-me-local-only" or empty strings are fine.
    for line in text.splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        _, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if not value or "change-me" in value:
            continue
        # If we reach here the example contains a non-placeholder value; fail loud.
        if value.startswith("sk-"):
            pytest.fail(f".env.example contains an OpenAI-style key: {line}")


def test_otel_collector_config_exists_and_is_yaml() -> None:
    cfg = ROOT / "deploy" / "docker" / "otel" / "collector-config.yaml"
    assert cfg.exists()
    text = cfg.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    assert "receivers" in parsed
    assert "exporters" in parsed
    assert "service" in parsed


def test_prometheus_config_exists_and_is_yaml() -> None:
    cfg = ROOT / "deploy" / "prometheus" / "prometheus.yml"
    assert cfg.exists()
    parsed = _yaml_load(cfg.read_text(encoding="utf-8"))
    assert "scrape_configs" in parsed


def test_grafana_dashboards_provisioning_exists() -> None:
    base = ROOT / "deploy" / "grafana"
    assert (base / "provisioning" / "datasources" / "prometheus.yml").exists()
    assert (base / "provisioning" / "dashboards" / "dashboards.yml").exists()
    assert (base / "dashboards" / "operations.json").exists()
