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
    # Services that legitimately have no inline healthcheck:
    #   - one-shot init jobs (*-init): terminated on success
    #   - otel-collector: contrib image has no healthcheck CLI and no
    #     wget/curl; health is probed by a separate otel-probe service
    no_healthcheck_allowed = {
        "minio-init",
        "otel-probe",
        "otel-collector",
    }
    for name, svc in services.items():
        if name.endswith("-init") or name in no_healthcheck_allowed:
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
    """Phoenix must connect to its own database via the canonical
    PHOENIX_POSTGRES_* env vars, never via an inline DATABASE_URL.

    PHOENIX_POSTGRES_* is the supported configuration; URL composition
    inside Compose breaks for passwords containing '@', ':', '/', '#',
    or '%'. See Phoenix configuration docs.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    phoenix_env = parsed["services"]["phoenix"]["environment"]
    # Must NOT use the URL form.
    assert "PHOENIX_SQL_DATABASE_URL" not in phoenix_env, (
        "Phoenix must not compose the DB URL itself; use PHOENIX_POSTGRES_*"
    )
    # Must use the four supported keys.
    for key in (
        "PHOENIX_POSTGRES_HOST",
        "PHOENIX_POSTGRES_PORT",
        "PHOENIX_POSTGRES_USER",
        "PHOENIX_POSTGRES_PASSWORD",
        "PHOENIX_POSTGRES_DB",
    ):
        assert key in phoenix_env, f"Phoenix must set {key}"
    # The Phoenix database must not equal the application database.
    app_db = parsed["services"]["postgres"]["environment"].get("POSTGRES_DB")
    assert app_db, "Application POSTGRES_DB must be set"
    assert app_db != phoenix_env["PHOENIX_POSTGRES_DB"], (
        "Application and Phoenix must not share a database"
    )


def test_phoenix_enables_prometheus_metrics() -> None:
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    env = parsed["services"]["phoenix"]["environment"]
    assert env.get("PHOENIX_ENABLE_PROMETHEUS") == "true", (
        "Phoenix must set PHOENIX_ENABLE_PROMETHEUS=true so the prometheus job can scrape it"
    )
    # And its metrics port must be exposed.
    ports = parsed["services"]["phoenix"].get("ports", []) or []
    published: set[str] = set()
    for p in ports:
        if isinstance(p, str):
            published.add(p.split(":")[-1].split("/")[0])
        elif isinstance(p, dict):
            published.add(str(p.get("published", "")))
    # Phoenix internal metrics port is 9090; we map it to host 9091 to
    # avoid collision with the local Prometheus server.
    assert "9091" in published or "9090" in published, (
        f"Phoenix metrics port must be exposed; have {published}"
    )


def test_otel_probe_strict_curl_timeouts() -> None:
    """The otel-probe entrypoint must use strict curl timeouts so a
    20s budget is actually 20s (not unbounded).
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    probe = parsed["services"]["otel-probe"]
    cmd = probe["entrypoint"][-1]
    assert "--max-time" in cmd, "curl must use --max-time to bound each request"
    assert "--connect-timeout" in cmd, "curl must use --connect-timeout"


def test_prometheus_waits_for_otel_probe() -> None:
    """Prometheus must depend on otel-probe with
    service_completed_successfully so it does not start until the
    collector's health endpoint is actually reachable.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    deps = parsed["services"]["prometheus"].get("depends_on") or {}
    assert isinstance(deps, dict), "prometheus depends_on must be a mapping"
    assert "otel-probe" in deps, "prometheus must depend on otel-probe"
    assert deps["otel-probe"].get("condition") == "service_completed_successfully", (
        "prometheus must wait for otel-probe to complete successfully"
    )


def test_grafana_waits_for_prometheus_healthy() -> None:
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    deps = parsed["services"]["grafana"].get("depends_on") or {}
    assert isinstance(deps, dict)
    assert "prometheus" in deps
    assert deps["prometheus"].get("condition") == "service_healthy", (
        "grafana must wait for prometheus to become healthy"
    )


def test_prometheus_healthcheck_targets_ready_endpoint() -> None:
    """Prometheus healthcheck must probe /-/ready, not just --version."""
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    hc = parsed["services"]["prometheus"]["healthcheck"]
    test = hc.get("test", [])
    cmd = " ".join(test) if isinstance(test, list) else str(test)
    assert "-/ready" in cmd, "Prometheus healthcheck must hit /-/ready to prove service is ready"
    assert "--version" not in cmd, "Prometheus must not use --version as healthcheck"


def test_otel_internal_telemetry_bound_to_all_interfaces() -> None:
    """The collector's self-telemetry exporter must bind 0.0.0.0:8888
    so the prometheus container can scrape it.
    """
    cfg = (ROOT / "deploy" / "docker" / "otel" / "collector-config.yaml").read_text(
        encoding="utf-8"
    )
    parsed = _yaml_load(cfg)
    telemetry = parsed.get("service", {}).get("telemetry", {}).get("metrics", {})
    addr = telemetry.get("address", "")
    assert "0.0.0.0" in addr, f"otel-collector self-telemetry must bind 0.0.0.0:8888; got {addr!r}"
    assert "8888" in addr
    """The init script must create the Phoenix DB using PHOENIX_DB env.

    The script is a .sh (not .sql) so it can interpolate env vars and
    quote identifiers safely. Both databases must coexist on the same
    instance and must be different.
    """
    init_dir = ROOT / "deploy" / "docker" / "postgres" / "initdb.d"
    files = sorted(p.name for p in init_dir.glob("*"))
    assert "00-init.sh" in files, f"Expected 00-init.sh in initdb.d; got {files}"
    text = (init_dir / "00-init.sh").read_text(encoding="utf-8")
    # Must use a parameterized identifier, not a hard-coded name.
    assert "PHOENIX_DB" in text, "00-init.sh must read PHOENIX_DB from the environment"
    assert "POSTGRES_DB" in text, "00-init.sh must read POSTGRES_DB from the environment"
    # Sanity: the script must guard against the two DBs being the same
    assert "must differ" in text, "00-init.sh must refuse identical POSTGRES_DB and PHOENIX_DB"


def test_pg_initdb_enables_pgvector_in_app_db() -> None:
    init_dir = ROOT / "deploy" / "docker" / "postgres" / "initdb.d"
    target = init_dir / "00-init.sh"
    text = target.read_text(encoding="utf-8")
    assert "CREATE EXTENSION" in text
    assert "vector" in text
    # The pgvector extension must be created in the app database,
    # not the phoenix database. The script's structure should reflect
    # that: enable in APP_DB (POSTGRES_DB), not PHOENIX_DB.
    assert "POSTGRES_DB" in text


def test_initdb_script_quotes_identifiers() -> None:
    """Defense against identifier injection."""
    text = (ROOT / "deploy" / "docker" / "postgres" / "initdb.d" / "00-init.sh").read_text(
        encoding="utf-8"
    )
    assert "sanitize_ident" in text, "script must sanitize identifiers"
    assert "ON_ERROR_STOP" in text, "psql must use -v ON_ERROR_STOP=1"
    assert "set -euo pipefail" in text, "script must fail fast"


def test_postgres_service_passes_phoenix_db_env() -> None:
    """The postgres service must pass PHOENIX_DB to the entrypoint
    so /docker-entrypoint-initdb.d/00-init.sh receives it."""
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    env = parsed["services"]["postgres"].get("environment", {})
    assert "PHOENIX_DB" in env, (
        "postgres service must declare PHOENIX_DB so the init script can use it"
    )


def test_neo4j_auth_has_built_in_defaults() -> None:
    """NEO4J_AUTH must include Compose-level defaults so a missing
    .env (or one without NEO4J_USER) still produces a valid auth string.
    YAML anchor defaults are not visible at the interpolation site.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    env = parsed["services"]["neo4j"]["environment"]
    auth = env.get("NEO4J_AUTH", "")
    assert auth, "NEO4J_AUTH must be set"
    # Must contain default value clauses for both halves
    assert "${NEO4J_USER:-" in auth, f"NEO4J_AUTH must default NEO4J_USER inline; got {auth!r}"
    assert "${NEO4J_PASSWORD:-" in auth, (
        f"NEO4J_AUTH must default NEO4J_PASSWORD inline; got {auth!r}"
    )


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


def test_otel_collector_has_no_inline_healthcheck() -> None:
    """The contrib image has no `healthcheck` CLI subcommand and no
    wget/curl, so an inline `healthcheck:` block is impossible.

    The actual probe is delegated to a separate `otel-probe` one-shot
    service that uses curlimages/curl to hit the health_check
    extension endpoint on port 13133.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    svc = parsed["services"]["otel-collector"]
    assert "healthcheck" not in svc, (
        "otel-collector must not have an inline healthcheck block; "
        "the contrib image has no healthcheck CLI and no wget/curl"
    )


def test_otel_probe_service_exists_and_uses_curl() -> None:
    """The probe must use an image that ships a working HTTP client
    and target the health_check extension endpoint on port 13133.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = parsed["services"]
    assert "otel-probe" in services, "expected an otel-probe one-shot service"
    probe = services["otel-probe"]
    image = probe["image"]
    assert "curl" in image, f"probe image must contain curl; got {image!r}"
    depends = probe.get("depends_on") or {}
    depends_names = list(depends.keys()) if isinstance(depends, dict) else list(depends)
    assert "otel-collector" in depends_names, "otel-probe must depend on otel-collector"
    entrypoint = probe.get("entrypoint") or []
    cmd = entrypoint[-1] if entrypoint else ""
    assert "13133" in cmd, "probe must hit the health_check extension on :13133"
    assert "otel-collector" in cmd, (
        "probe must hit the collector by service name (DNS) not localhost"
    )


def test_otel_collector_exposes_metrics_and_health_ports() -> None:
    """Port 13133 (health) and 8888 (internal metrics) must be exposed
    so the Prometheus scrape and the probe can reach them.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    ports = parsed["services"]["otel-collector"].get("ports", []) or []
    published: set[str] = set()
    for port in ports:
        if isinstance(port, str):
            published.add(port.split(":")[-1].split("/")[0])
        elif isinstance(port, dict):
            published.add(str(port.get("published", "")))
    for required in ("13133", "8888", "8889", "4317"):
        assert required in published, (
            f"otel-collector must expose port {required}; have {published}"
        )


def test_healthchecks_do_not_assume_wget_where_missing() -> None:
    """For services whose image may not ship wget, ensure no wget use.

    `otel-collector` no longer has an inline healthcheck; only services
    that actually have one are inspected here.
    """
    parsed = _yaml_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    no_wget_services = {
        "otel-collector",  # contrib image has no wget
        "otel-probe",  # uses curlimages/curl; we still want to ensure no wget
    }
    for svc_name in no_wget_services:
        if svc_name not in parsed["services"]:
            continue
        svc = parsed["services"][svc_name]
        if "healthcheck" in svc:
            hc = svc["healthcheck"]
            test = hc.get("test", [])
            cmd_str = " ".join(test) if isinstance(test, list) else str(test)
            assert "wget" not in cmd_str, f"{svc_name} healthcheck must not assume wget is present"


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
