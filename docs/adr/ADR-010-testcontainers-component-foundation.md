- **Status**: Accepted
- **Date**: 2026-09-05
- **Deciders**: implementation agent + user
- **Related**: AGENTS.md §7 (verification commands), M1 sign-off, M1 follow-up (this ADR), future M2 adapters

## Context

M1 sign-off established the full docker-compose stack as the integration
testing substrate (12 stack-smoke tests + 147 unit tests, all verified
locally against Docker). However, that setup is not the right shape
for the next 6+ milestones:

  1. **Cost per run** — bringing up all 7 services (Postgres, Neo4j,
     MinIO, OTel, Phoenix, Prometheus, Grafana) takes 60-90s. Running
     these for every single repo-level test would balloon CI time.
  2. **Granularity** — when a test fails, the failure could be in any
     service or in the wiring between them. Debugging is hard.
  3. **Fault isolation** — a future test that needs to stop Postgres
     mid-test should NOT take down Phoenix, Grafana, etc. The
     full-stack test can only do destructive things in one specific
     way (it can stop the postgres container, but the test only
     exists once).
  4. **Layer coverage** — M2+ adds application-level adapters
     (Postgres document repository, Neo4j knowledge graph, etc.).
     These adapters need to be tested against the real engine
     (Postgres 16, Neo4j 5.26) without paying the cost of the
     full observability stack.

The existing M1.4 stack tests are valuable as end-to-end readiness
evidence (network connectivity, port mapping, healthcheck wiring,
OTel/Phoenix export pipeline) and must be preserved.

## Decision

Adopt a **four-layer test split**, codified in pytest markers and
Makefile targets:

| Layer            | Marker                                | Make target        | Docker        | Cost / run |
|------------------|---------------------------------------|--------------------|---------------|------------|
| Unit             | (none)                                | `make test`        | no            | ~2 s       |
| Component        | `@pytest.mark.integration` + `@pytest.mark.component` | `make test-component` | Testcontainers (one container per dependency per session) | ~10 s |
| Stack smoke      | `@pytest.mark.integration` + `@pytest.mark.stack` | `make test-stack` | docker compose up (7 services) | ~90 s |
| Fault injection  | `@pytest.mark.integration` + `@pytest.mark.fault` | `make test-fault` (reserved for M11) | dedicated Testcontainer (or other isolated fixture) — destructive action runs in a per-test container, never the shared compose session | isolated per test |
| Combined         | `@pytest.mark.integration`            | `make test-integration` | both | ~100 s |

Rules:

  1. The `integration` marker is the umbrella. Sub-markers
     (`component`, `stack`, `fault`) MUST be combined with
     `integration` so the existing `make test` filter
     (`-m "not integration"`) keeps working unchanged.
  2. Component tests use **Testcontainers** with **session-scoped
     fixtures** — one Postgres container and one Neo4j container
     are started per pytest session, not per test function. Image
     versions are pinned to the exact tags used in docker-compose
     (`pgvector/pgvector:0.8.6-pg16`, `neo4j:5.26.10-community`).
  3. Stack smoke tests retain the existing M1 fixtures and
     container IDs are NOT to be widened further. The single
     `TestDestructivePostgresRecovery` test stays as the M1
     readiness evidence.
  4. The `testcontainers` package is pinned to a specific version
     (`==4.15.0`) so `uv.lock` is the single source of truth.
  5. MinIO is deferred to M3; the `minio` extra is dropped from
     the testcontainers extra list until M3 needs it.

## Consequences

Easier:

  * M2+ repository/adapter tests can run in ~10s without the
    observability stack. CI can run `make test-component` on
    every PR and `make test-stack` only on merge to main.
  * Fault-injection tests (M11 territory per plan §0.3) get
    their own `fault` marker, isolated from `component` so a
    destructive test can never pollute the cheap component suite.
  * When a component test fails, the failure is unambiguous —
    it is the substrate (Postgres, Neo4j) or the adapter, not
    some cross-service wiring.
  * `make test-component` is hermetic — no shared state, no
    port collisions (Testcontainers uses dynamic ports).

Harder:

  * Two infra stacks to maintain: docker-compose for `stack`
    tests, Testcontainers for `component` tests. Image
    versions MUST be kept in sync (mitigated by pinning
    both to the same string in `tests/component/conftest.py`).
  * New devs must understand the marker layering.

Trade-offs accepted:

  * The full `make test-integration` is slower than before
    (component + stack), but `make test-component` alone
    covers most of what devs need locally.

## Alternatives considered

  1. **Keep docker-compose as the only infra substrate.**
     Rejected: cost per run is too high for M2+ where adapter
     tests will multiply.

  2. **Use Testcontainers for everything, retire docker-compose.**
     Rejected: loses the end-to-end readiness evidence (Phoenix
     export pipeline, Grafana dashboard, network topology).
     The user explicitly required keeping the M1.4 stack
     tests as a minimal acceptance evidence.

  3. **Use ephemeral containers in Python for unit tests.**
     Rejected: cross-service wiring is what we need to verify
     in stack tests; an ephemeral in-process container is
     not the same artifact as the production image.

## References

  * `tests/component/conftest.py` — Testcontainers fixtures
  * `tests/component/test_postgres.py` — 4 component tests
  * `tests/component/test_neo4j.py` — 4 component tests
  * `pyproject.toml` — `[tool.pytest.ini_options].markers`
  * `Makefile` — `test`, `test-component`, `test-stack`,
    `test-fault`, `test-integration`
  * `uv.lock` — `testcontainers==4.15.0` (single source of
    truth for the dep version)
  * `docs/plan.md` — M1 follow-up note on the progress ledger
    (this work is part of M1 follow-up, not a separate milestone)
