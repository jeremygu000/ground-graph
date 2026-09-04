"""Local infrastructure validation tests.

These tests verify the docker-compose configuration is well-formed and
that the listed services match the dependency requirements in plan.md §M1.
The tests do NOT require Docker to run; they only need the file to be valid YAML.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
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
        "prometheus-data",
        "grafana-data",
    }
    missing = sorted(expected - set(volumes.keys()))
    assert not missing, f"Missing named volumes: {missing}"


def test_docker_compose_no_fixed_container_names() -> None:
    """container_name breaks isolation between checkouts and CI jobs.

    Per the M1 review: all services should rely on the Compose project
    name for isolation, not a hard-coded container_name.
    """
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "container_name:" not in text, (
        "docker-compose.yml must not set container_name; use the project name"
    )
    parsed = _yaml_load(text)
    for vol_name, vol_cfg in (parsed.get("volumes") or {}).items():
        cfg = vol_cfg or {}
        assert not cfg.get("name"), f"Volume {vol_name!r} must not pin a name"
    for net_name, net_cfg in (parsed.get("networks") or {}).items():
        cfg = net_cfg or {}
        assert not cfg.get("name"), f"Network {net_name!r} must not pin a name"


def test_docker_compose_project_name_is_groundgraph() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    assert parsed.get("name") == "groundgraph", (
        f"Compose project name must be 'groundgraph', got {parsed.get('name')!r}"
    )


def test_docker_compose_all_ports_bind_localhost_only() -> None:
    """Local dev must not expose services on 0.0.0.0; bind to 127.0.0.1."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    # Find every short-form port mapping "X:Y" or "X:Y:Z" and assert
    # either a full "127.0.0.1:X:Y" form or an explicit "127.0.0.1:...".
    pattern = re.compile(r'^\s*-\s*"([^"]+)"\s*$', re.MULTILINE)
    bad: list[str] = []
    for svc_name, svc in parsed.get("services", {}).items():
        for port in svc.get("ports", []) or []:
            if isinstance(port, str):
                line = port
            elif isinstance(port, dict):
                line = f"{port.get('published', '')}:{port.get('target', '')}"
            else:
                continue
            # Strip /protocol suffix
            line = line.split("/")[0]
            if "127.0.0.1" not in line:
                bad.append(f"{svc_name}: {line!r}")
    assert not bad, f"All ports must be bound to 127.0.0.1 (localhost only). Found: {bad}"
    # Sanity: at least one port mapping exists
    assert pattern.findall(text), "Expected at least one port mapping"


def test_phoenix_uses_separate_database() -> None:
    """Phoenix must connect to its own database, not the application one."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    phoenix_env = parsed["services"]["phoenix"]["environment"]
    # find the SQL url
    sql_url = ""
    for k, v in phoenix_env.items():
        if k == "PHOENIX_SQL_DATABASE_URL":
            sql_url = v
    assert sql_url, "Phoenix must set PHOENIX_SQL_DATABASE_URL"
    # The variable form ${PHOENIX_DB:-phoenix} resolves to 'phoenix';
    # the default branch in the value string should reference phoenix too.
    assert "phoenix" in sql_url.lower(), (
        f"Phoenix URL must point at a phoenix database; got {sql_url!r}"
    )
    # The application database must not be phoenix
    app_db = parsed["services"]["postgres"]["environment"].get("POSTGRES_DB")
    assert app_db, "Application POSTGRES_DB must be set"
    assert app_db != "phoenix", "Application and Phoenix must not share a database"


def test_pg_initdb_creates_separate_phoenix_database() -> None:
    init_dir = ROOT / "deploy" / "docker" / "postgres" / "initdb.d"
    files = sorted(p.name for p in init_dir.glob("*.sql"))
    assert "00-create-phoenix-db.sql" in files, (
        f"Expected 00-create-phoenix-db.sql in initdb.d; got {files}"
    )
    sql = (init_dir / "00-create-phoenix-db.sql").read_text(encoding="utf-8")
    assert "CREATE DATABASE phoenix" in sql, (
        "00-create-phoenix-db.sql must create the phoenix database"
    )


def test_pg_initdb_enables_pgvector_in_app_db() -> None:
    init_dir = (
        ROOT / "deploy" / "postgres" / "initdb.d"
        if (ROOT / "deploy" / "postgres" / "initdb.d").exists()
        else ROOT / "deploy" / "docker" / "postgres" / "initdb.d"
    )
    target = init_dir / "10-app-extensions.sql"
    assert target.exists(), f"Expected {target}"
    text = target.read_text(encoding="utf-8")
    assert "CREATE EXTENSION" in text
    assert "vector" in text
    assert "\\connect groundgraph" in text or "groundgraph" in text


def test_neo4j_has_no_apoc_configuration() -> None:
    """MVP does not require APOC; APOC references must be removed entirely."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    parsed = _yaml_load(text)
    env = parsed["services"]["neo4j"]["environment"]
    assert env.get("NEO4J_PLUGINS") == "[]", (
        "Neo4j plugins list must be empty '[]' unless APOC is actually needed"
    )
    for k in env:
        assert "apoc" not in k.lower(), (
            f"Neo4j must not reference apoc when no plugin is installed; found {k}"
        )
    assert "apoc" not in text.lower(), (
        "docker-compose.yml must not mention apoc when no plugin is installed"
    )


def test_minio_init_uses_strict_mode_and_ignore_existing() -> None:
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    init_svc = parsed["services"]["minio-init"]
    entrypoint = init_svc["entrypoint"]
    assert isinstance(entrypoint, list), "minio-init entrypoint must be a list"
    assert entrypoint[:2] == ["/bin/sh", "-ec"], (
        f"minio-init must use /bin/sh -ec; got {entrypoint!r}"
    )
    cmd = entrypoint[2]
    assert "|| true" not in cmd, "minio-init must not swallow errors with || true"
    assert "--ignore-existing" in cmd, "minio-init must use --ignore-existing to be idempotent"


def test_no_unused_grafana_plugins() -> None:
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    env = parsed["services"]["grafana"]["environment"]
    assert "GF_INSTALL_PLUGINS" not in env, (
        "Grafana plugins should be removed; pin per-need rather than install-by-default"
    )


def test_otel_collector_does_not_depend_on_postgres_or_neo4j() -> None:
    """The collector exports to Phoenix/Prometheus, not to PG/Neo4j directly."""
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    deps = parsed["services"]["otel-collector"].get("depends_on") or {}
    dep_names = list(deps.keys()) if isinstance(deps, dict) else list(deps)
    assert "postgres" not in dep_names
    assert "neo4j" not in dep_names


def test_otel_collector_healthcheck_uses_local_binary() -> None:
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    hc = parsed["services"]["otel-collector"]["healthcheck"]
    test = hc.get("test", [])
    # The contrib image ships /otelcol-contrib with a healthcheck subcommand
    cmd_str = " ".join(test) if isinstance(test, list) else str(test)
    assert "wget" not in cmd_str, (
        "OTel collector image does not include wget; use the built-in healthcheck subcommand"
    )
    assert "healthcheck" in cmd_str


def test_healthchecks_do_not_assume_wget_where_missing() -> None:
    """For services whose image may not ship wget, ensure no wget use."""
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    no_wget_images = {
        "otel-collector": "otel/opentelemetry-collector-contrib",
    }
    for svc_name, image in no_wget_images.items():
        hc = parsed["services"][svc_name]["healthcheck"]
        test = hc.get("test", [])
        cmd_str = " ".join(test) if isinstance(test, list) else str(test)
        assert "wget" not in cmd_str, (
            f"{svc_name} ({image}) healthcheck must not assume wget is present"
        )


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
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
        fh.write("NEO4J_USER=neo4j\n")
        fh.write("NEO4J_PASSWORD=change-me-local-only\n")
        fh.write("POSTGRES_USER=groundgraph\n")
        fh.write("POSTGRES_PASSWORD=change-me-local-only\n")
        fh.write("PHOENIX_DB=phoenix\n")
        env_path = fh.name
    try:
        proc = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "--env-file",
                env_path,
                "config",
                "-q",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(env_path).unlink()
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        if "cannot connect" in out.lower() or "docker daemon" in out.lower():
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
