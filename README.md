# groundgraph

Hybrid Agentic GraphRAG system combining a vertical knowledge graph, a horizontal workflow graph, and a queryable execution graph. Authoritative plan: `docs/plan.md`. Executable contract: `AGENTS.md`.

## Status

| Milestone | Status |
|---|---|
| M0 — Repository and engineering baseline | Complete |
| M1 — Local infrastructure and telemetry | Complete |
| M2 — Domain contracts and persistence | Pending |
| M3 — Document ingestion and versioning | Pending |
| M4 — Vector RAG baseline | Pending |
| M5 — Knowledge graph construction | Pending |
| M6 — Hybrid GraphRAG retrieval | Pending |
| M7 — Query workflow, citations, and API | Pending |
| M8 — Evaluation system and CI gates | Pending |
| M9 — Governance, security, adversarial | Pending |
| M10 — Operator and review interfaces | Pending |
| M11 — Production hardening | Pending |
| M12 — Controlled improvement loop | Pending |

## Stack

Python 3.12 · uv · FastAPI · Pydantic v2 · SQLAlchemy 2 async · Alembic · PostgreSQL + pgvector · Neo4j · LangGraph · OpenAI SDK · MinIO · OpenTelemetry/OTLP · Phoenix · Prometheus · Grafana · Ruff · Pyright · pytest.

## Quick start

```bash
make setup          # install uv deps, pre-commit
make format         # ruff format
make lint           # ruff check
make typecheck      # pyright
make test           # pytest (unit only)
make test-integration  # bring up docker-compose stack and run integration tests
make check          # format + lint + typecheck + test
```

## Local infrastructure

M1 brings up a 7-service local stack via docker-compose, with per-service
memory caps so the full stack fits comfortably in 7-8 GB of host RAM
(observed RSS < 1.5 GB):

```bash
docker compose up -d
```

| Service | Port (host) | Purpose |
|---|---|---|
| PostgreSQL + pgvector | 5432 | Application database |
| Neo4j | 7474 / 7687 | Knowledge graph |
| MinIO | 9000 / 9001 | S3-compatible object storage |
| OTel Collector | 4317 / 4318 / 8888 / 8889 | OTLP ingest, Prometheus exporter |
| Phoenix | 6006 | Trace UI + observability backend |
| Prometheus | 9090 | Metrics scraper + storage |
| Grafana | 3001 | Dashboards (admin / `change-me-local-only`) |

The FastAPI app connects to the stack at runtime via the
`app_process` pytest fixture (or manually via the same env vars).
Health endpoints:

- `GET /health/live` — process liveness, always 200
- `GET /health/ready` — 200 when all dependencies are reachable, 503 otherwise

## Layout

```
apps/                    # api, ingestion_worker, evaluation_runner, web
  src/groundgraph/
  domain/                # pure types & policies (no framework imports)
  application/           # use cases & ports
  workflows/             # LangGraph adapter (calls application)
  infrastructure/        # adapters: postgres, neo4j, openai, otel, s3
  api/                   # FastAPI routers
ontology/                # versioned YAML ontology
evals/                   # datasets, metrics, judges, reports
migrations/              # Alembic
tests/                   # unit, integration, contract, end_to_end, adversarial
deploy/                  # docker, prometheus, grafana, phoenix
docs/                    # adr, api, operations, evaluation
```

## Architecture boundary

```
domain  <-  application  <-  workflows / API  <-  infrastructure composition
```

Application/domain layers must not import LangGraph, FastAPI, OpenAI, Neo4j, SQLAlchemy, or Phoenix. See `AGENTS.md` §4 and `docs/adr/ADR-001-python-clean-architecture.md`.

## See also

- `docs/plan.md` — authoritative implementation plan
- `AGENTS.md` — execution contract for coding agents
- `docs/adr/` — architecture decision records
