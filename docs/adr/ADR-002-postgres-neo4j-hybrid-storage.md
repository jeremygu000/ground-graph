# ADR-002 PostgreSQL/pgvector + Neo4j Hybrid Storage

- **Status**: Accepted
- **Date**: 2026-09-04
- **Deciders**: groundgraph team
- **Related**: ADR-001 (Python + clean architecture), ADR-005 (OTel content capture), plan.md §5

## Context

Groundgraph requires three distinct storage concerns that no single database optimally serves:

1. **Structured relational data** — documents, chunks, run records, execution audit trails, ACL metadata. PostgreSQL is the natural fit.
2. **Vector similarity search** — embedding-based retrieval over chunk content. pgvector (PostgreSQL extension) provides this without a separate vector store.
3. **Property-graph traversal** — multi-hop entity resolution, fact provenance chains, ontological constraints. Neo4j's Cypher query language and native graph engine are purpose-built for this.

Each concern has different access patterns, consistency requirements, and tooling ecosystems.

## Decision

We run two database technologies in parallel:

| Concern | Store | Rationale |
|---|---|---|
| Documents, chunks, run state, ACL | PostgreSQL | ACID transactions, Alembic migrations, rich SQL |
| Chunk embeddings | pgvector (same PostgreSQL instance) | Co-located with chunks; no extra service |
| Entities, facts, provenance, ontological constraints | Neo4j | Native graph engine, Cypher, schema constraints |

PostgreSQL is the source of truth for document ingestion status and run state. Neo4j is the source of truth for graph entities and facts. The two are kept eventually consistent via an outbox pattern (plan.md §5.3).

## Consequences

- **Easier**: Each concern uses the right tool; pgvector avoids a separate vector store; Neo4j Cypher is expressive for graph traversal.
- **Harder**: Two database technologies means two migration systems, two connection pools, two health checks, and reconciliation logic between stores.
- **Trade-offs accepted**: We deliberately exclude distributed ACID transactions between PostgreSQL and Neo4j. Consistency is achieved via outbox + idempotent projection, not two-phase commit.

## Alternatives considered

- **Single Neo4j with APOC for vectors**: Neo4j has vector search (晚8+), but it is not the primary use case; pgvector is more mature and better integrated with PostgreSQL for hybrid SQL+vector queries.
- **Single PostgreSQL with Neo4j foreign data wrapper**: FDW support for Neo4j is immature; Cypher queries over FDW lose expressiveness.
- **Separate vector store (Pinecone, Weaviate, Qdrant)**: Adds a third infrastructure dependency; pgvector is sufficient for the expected dataset scale.
- **Neo4j for everything (including documents)**: Neo4j is not designed for high-volume document/chunk storage or ACID run-state tracking.

## References

- plan.md §5 (storage architecture)
- plan.md §5.1 (PostgreSQL tables)
- plan.md §5.2 (Neo4j constraints and indexes)
- plan.md §5.3 (consistency between PostgreSQL and Neo4j)
- ADR-001 (Python + clean architecture)
