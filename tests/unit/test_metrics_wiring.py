"""M1.2 — three-way Prometheus metrics wiring tests.

Verifies that:
  - prometheus.yml defines three independent scrape jobs
  - the OTel internal telemetry bind address is reachable
  - the Grafana operations dashboard references the new job names
  - all target endpoints (8888, 8889, 9090) appear in the config
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROM_YML = ROOT / "deploy" / "prometheus" / "prometheus.yml"
COLLECTOR_CFG = ROOT / "deploy" / "docker" / "otel" / "collector-config.yaml"
DASHBOARD = ROOT / "deploy" / "grafana" / "dashboards" / "operations.json"
COMPOSE = ROOT / "docker-compose.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _collect_targets(prom: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for job in prom.get("scrape_configs", []) or []:
        targets: list[str] = []
        for sc in job.get("static_configs", []) or []:
            for t in sc.get("targets", []) or []:
                targets.append(t)
        out[job.get("job_name", "<missing>")] = targets
    return out


def test_prometheus_has_three_jobs() -> None:
    """The plan and M1.2 require three independent scrape jobs."""
    prom = _load_yaml(PROM_YML)
    jobs = set(_collect_targets(prom).keys())
    required = {
        "otel-collector-internal",
        "groundgraph-application",
        "phoenix",
    }
    missing = required - jobs
    assert not missing, f"Missing Prometheus jobs: {missing}; have {jobs}"


def test_otel_internal_targets_collector_8888() -> None:
    prom = _load_yaml(PROM_YML)
    targets = _collect_targets(prom).get("otel-collector-internal", [])
    assert "otel-collector:8888" in targets, (
        f"otel-collector-internal must scrape :8888; have {targets}"
    )


def test_groundgraph_application_targets_collector_8889() -> None:
    prom = _load_yaml(PROM_YML)
    targets = _collect_targets(prom).get("groundgraph-application", [])
    assert "otel-collector:8889" in targets, (
        f"groundgraph-application must scrape :8889; have {targets}"
    )


def test_phoenix_job_targets_phoenix_9090() -> None:
    prom = _load_yaml(PROM_YML)
    targets = _collect_targets(prom).get("phoenix", [])
    assert "phoenix:9090" in targets, f"phoenix job must scrape :9090; have {targets}"


def test_collector_self_telemetry_bound_to_all_interfaces() -> None:
    """Without 0.0.0.0:8888, the prometheus container cannot scrape."""
    cfg = _load_yaml(COLLECTOR_CFG)
    metrics = cfg.get("service", {}).get("telemetry", {}).get("metrics", {})
    addr = metrics.get("address", "")
    assert "0.0.0.0" in addr, f"collector self-telemetry must bind 0.0.0.0; got {addr!r}"
    assert "8888" in addr


def test_collector_has_prometheus_exporter_on_8889() -> None:
    cfg = _load_yaml(COLLECTOR_CFG)
    exporters = cfg.get("exporters", {})
    prom_exp = exporters.get("prometheus", {})
    endpoint = prom_exp.get("endpoint", "")
    assert "0.0.0.0:8889" in endpoint, (
        f"prometheus exporter must bind 0.0.0.0:8889; got {endpoint!r}"
    )


def test_dashboard_references_otel_internal_job() -> None:
    """Grafana queries must use the otel-collector-internal job label,
    not the collector-internal exporter port (:8889).
    """
    text = DASHBOARD.read_text(encoding="utf-8")
    assert "otel-collector-internal" in text, (
        "Grafana dashboard must reference the 'otel-collector-internal' job"
    )
    # And must NOT query metrics from :8889 (that is the application job)
    # The dashboard's app panel can reference 8889 in the title but the
    # otelcol_* metrics it queries must be on the internal job.
    parsed = json.loads(text)
    for panel in parsed.get("panels", []) or []:
        for target in panel.get("targets", []) or []:
            expr = target.get("expr", "")
            if "otelcol_" in expr:
                assert 'job="otel-collector-internal"' in expr or ('job=~"$job"' in expr), (
                    f"otelcol_* query must be filtered to otel-collector-internal; got {expr!r}"
                )


def test_dashboard_references_groundgraph_application_job() -> None:
    """The application connectivity panel must point at the application job."""
    parsed = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    queries = []
    for panel in parsed.get("panels", []) or []:
        for target in panel.get("targets", []) or []:
            expr = target.get("expr", "")
            if isinstance(expr, str):
                queries.append(expr)
    joined = "\n".join(queries)
    assert 'job="groundgraph-application"' in joined
    assert 'up{job="groundgraph-application"}' in joined


def test_phoenix_prometheus_metrics_are_enabled() -> None:
    """Phoenix must expose Prometheus metrics for the in-network scrape."""
    compose = _load_yaml(COMPOSE)
    env = compose["services"]["phoenix"].get("environment", {})
    assert env.get("PHOENIX_ENABLE_PROMETHEUS") == "true"


def test_no_port_collision_on_9090() -> None:
    """Host port 9090 must not be used by both prometheus and phoenix."""
    compose = _load_yaml(COMPOSE)

    def _host_ports(svc: dict[str, Any]) -> set[str]:
        out: set[str] = set()
        for p in svc.get("ports", []) or []:
            if isinstance(p, str):
                # "127.0.0.1:9090:9090" -> first segment is the host port
                segs = p.split(":")
                if len(segs) >= 2:
                    out.add(segs[1].split("/")[0])
            elif isinstance(p, dict):
                if p.get("published"):
                    out.add(str(p["published"]))
        return out

    prom_ports = _host_ports(compose["services"]["prometheus"])
    phoenix_ports = _host_ports(compose["services"]["phoenix"])
    assert prom_ports.isdisjoint(phoenix_ports), (
        f"host ports must not collide: prometheus={prom_ports}, phoenix={phoenix_ports}"
    )
