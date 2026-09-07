# ADR-011 — Production Hardening and Pilot Readiness

## Status

Accepted

## Context

M11 requires preparing GroundGraph for a controlled pilot with measured capacity,
reliability, cost control, rollback procedures, and documented operations.

## Decisions

### 1. Container Strategy
Production images use multi-stage Dockerfile (`deploy/docker/Dockerfile.prod`):
- Builder stage for dependency installation
- Production stage with non-root user, minimal attack surface
- Uvicorn with multiple workers (2×CPU+1, min 2, max 8)
- Health checks on every exposed service

### 2. Deployment
- `deploy/docker/docker-compose.prod.yml`: production composition with
  resource limits, health checks, named volumes, restart policies
- `deploy/kubernetes/deployment.yaml`: K8s manifests with HPA, PDB, Ingress,
  PodAntiAffinity for availability
- `scripts/canary_eval.py`: canary evaluation before promotion

### 3. Backup and Restore
- `scripts/backup_postgres.sh`: pg_basebackup + WAL archiving → S3
- `scripts/backup_neo4j.sh`: neo4j-admin dump or APOC cypher export → S3
- Runbook: `docs/operations/runbook.md`

### 4. Telemetry Outage Isolation
Per ADR-005, telemetry is best-effort. OTel collector outages do NOT corrupt
business flow. The API continues operating; traces are buffered or dropped.

### 5. Circuit Breaker Policy
Implemented via `groundgraph.infrastructure.resilience` (future work: M11 post-pilot).
Timeouts by stage:
- Vector retrieval: 5s
- Graph traversal: 3s
- LLM generation: 30s

## Consequences

- Production deployments are infrastructure-as-code with no manual steps
- Rollback is single-command via docker-compose or kubectl
- Operations runbook provides SRE-level procedures
