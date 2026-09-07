# GroundGraph Operations Runbook

> Operational procedures for the GroundGraph pilot. Covers common scenarios,
> incident response, backup/restore, and deployment operations.

---

## 1. Service Overview

| Service | Port | Description |
|---|---|---|
| `groundgraph_api` | 8000 | FastAPI application server |
| `postgres` | 5432 | Primary relational store |
| `pgvector` | 5433 | Vector similarity search |
| `neo4j` | 7687 (Bolt), 7474 (HTTP) | Graph database |
| `minio` | 9000 (API), 9001 (Console) | S3-compatible object storage |
| `prometheus` | 9090 | Metrics collection |
| `otel-collector` | 4317 (gRPC), 4318 (HTTP) | OTel trace/metric collection |

---

## 2. Common Operations

### 2.1 Deploy a New Version

```bash
# Build and push new image
docker build -f deploy/docker/Dockerfile.prod -t groundgraph/api:$VERSION .
docker push groundgraph/api:$VERSION

# Rolling update via docker-compose (production)
IMAGE_TAG=$VERSION docker compose -f deploy/docker/docker-compose.prod.yml up -d --no-deps groundgraph_api

# Verify rollout
curl -f http://localhost:8000/health/ready
```

### 2.2 Check System Health

```bash
# Full health check
curl -s http://localhost:8000/health/full | jq .

# Individual dependency health
curl -s http://localhost:8000/health/ready | jq .

# Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets'
```

### 2.3 View Logs

```bash
# API logs
docker compose logs -f groundgraph_api

# All services
docker compose logs -f

# Filter by trace ID
docker compose logs groundgraph_api | grep "trace_id=abc123"
```

---

## 3. Backup and Restore

### 3.1 PostgreSQL Backup

```bash
# Manual backup to S3
S3_BUCKET_BACKUP=groundgraph-backups \
POSTGRES_HOST=postgres \
POSTGRES_USER=groundgraph \
POSTGRES_DB=groundgraph \
./scripts/backup_postgres.sh

# Restore from S3
aws s3 cp s3://groundgraph-backups/postgres/YYYYMMDDTHHMMSSZ/base.tar.gz /tmp/
tar -xzf /tmp/base.tar.gz -C /var/lib/postgresql/data/
docker compose restart postgres
```

### 3.2 Neo4j Backup

```bash
# Manual Neo4j backup
NEO4J_AUTH=neo4j/<password> ./scripts/backup_neo4j.sh

# Restore
aws s3 cp s3://groundgraph-backups/neo4j/YYYYMMDDTHHMMSSZ/neo4j-backup.tar.gz /tmp/
tar -xzf /tmp/neo4j-backup.tar.gz -C /tmp/
docker compose exec neo4j neo4j-admin database load neo4j --from-path=/tmp/dumps --overwrite=true
docker compose restart neo4j
```

---

## 4. Incident Response

### 4.1 High Error Rate (5xx)

1. Check API health: `curl http://localhost:8000/health/full`
2. Check dependency health (postgres, neo4j, pgvector)
3. Check memory/CPU: `docker stats --no-stream`
4. Review recent logs: `docker compose logs --since=30m groundgraph_api`
5. Scale up if needed: `docker compose up -d --scale groundgraph_api=3`

### 4.2 Neo4j Unavailable

1. Check Neo4j status: `docker compose exec neo4j neo4j status`
2. Check disk space: `docker compose exec neo4j df -h`
3. If disk full, clear old transaction logs in `deploy/docker/neo4j/plugins/`
4. Restart: `docker compose restart neo4j`
5. Verify: `curl http://localhost:7474`

### 4.3 Vector Retrieval Degraded

1. Check pgvector health: `docker compose exec pgvector pg_isready`
2. Check disk I/O: `docker stats --no-stream`
3. Rebuild index if corrupted:
   ```bash
   docker compose exec pgvector psql -U groundgraph -d groundgraph \
     -c "REINDEX INDEX CONCURRENTLY idx_chunk_embedding;"
   ```

### 4.4 Telemetry Outage

OTel collector outages do NOT affect core business logic. Business flow continues.
To recover telemetry:
1. Restart collector: `docker compose restart otel-collector`
2. Replay buffered spans from Phoenix UI if available

---

## 5. Rollback Procedures

### 5.1 API Rollback

```bash
# Immediate rollback to previous image
docker compose pull groundgraph_api
docker compose up -d --no-deps groundgraph_api

# Full rollback via docker-compose.prod.yml
git checkout tags/v<previous> -- deploy/docker/docker-compose.prod.yml
docker compose -f deploy/docker/docker-compose.prod.yml up -d --no-deps groundgraph_api
```

### 5.2 Database Migration Rollback

```bash
# Rollback last Alembic revision
docker compose exec groundgraph_api alembic downgrade -1

# Verify
docker compose exec postgres psql -U groundgraph -d groundgraph -c "SELECT version_num FROM alembic_version;"
```

---

## 6. Performance Tuning

### 6.1 Query Latency Budget

| Stage | Target p99 |
|---|---|
| Auth + tenant resolution | < 5ms |
| Vector retrieval | < 100ms |
| Graph traversal | < 50ms |
| LLM answer generation | < 2000ms |
| **Total** | **< 3000ms** |

### 6.2 Resource Scaling

| Workload | API replicas | Postgres memory | Neo4j memory |
|---|---|---|---|
| Pilot (<100 QPS) | 2 | 2 GB | 2 GB |
| Growth (100-500 QPS) | 4 | 4 GB | 4 GB |
| Production (>500 QPS) | 8+ | 8 GB | 8 GB |

---

## 7. Evaluation and Monitoring

### 7.1 Run Evaluation Suite

```bash
make eval-smoke                    # Smoke tests
uv run pytest tests/integration/  # Full stack tests (requires Docker)
uv run python scripts/canary_eval.py --stable-url ... --canary-url ... --promote
```

### 7.2 Check Evaluation Metrics

```bash
# Prometheus metrics
curl -s http://localhost:9090/api/v1/query?query=groundgraph_evaluation_score | jq .

# Check SLOs
grep -E "SLO|SLI" docs/plan.md
```
