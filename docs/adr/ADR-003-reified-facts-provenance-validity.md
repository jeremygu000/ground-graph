# ADR-003 Reified Facts with Provenance and Validity Intervals

> **Status**: Accepted
> **Date**: 2026-09-05
> **Deciders**: Plan author
> **Related**: ADR-002, plan.md §3.3, §3.4, §4.2, §5.3

## Context

Knowledge-graph relationships (edges) in a naive GraphRAG suffer from three problems:

1. **No provenance** — an edge "A DEPENDS_ON B" has no record of *why* the system believes this or *what* evidence backs it.
2. **No temporal model** — relationships are true "now" but the graph has no mechanism to say "this was true from 2024-01 to 2024-06 but is no longer true".
3. **No confidence** — LLM-extracted facts have unknown reliability but are stored identically to structured facts.

Without reification, a retrieval that traverses a path like `(Service A) -[:DEPENDS_ON]-> (Service B) -[:DEPENDS_ON]-> (Queue C)` cannot distinguish "A relies on B because we parsed the ADR" from "A relies on B because an LLM hallucinated it". A query asking "what was true about Service A in March 2024" has no way to express temporal filtering.

## Decision

We adopt **reified facts** as the authoritative representation in Neo4j:

```cypher
(subject:Entity)-[:SUBJECT_OF]->(fact:Fact)-[:OBJECT]->(object:Entity)
(fact)-[:SUPPORTED_BY]->(chunk:Chunk)
(fact)-[:DEFINED_BY]->(document:DocumentVersion)
```

Every `Fact` node carries:

| Field | Type | Purpose |
|---|---|---|
| `fact_id` | UUID | Primary key |
| `predicate` | string | The relationship type (DEPENDS_ON, CALLS, etc.) |
| `status` | enum | `candidate` \| `verified` \| `rejected` \| `superseded` |
| `confidence` | float [0,1] | Extraction reliability |
| `evidence_ids` | list[UUID] | Chunks backing the fact |
| `valid_from` | datetime\|null | When the fact became true |
| `valid_to` | datetime\|null | When the fact stopped being true (null = still true) |
| `observed_at` | datetime | When this system learned the fact |
| `extraction_method` | enum | `structured` \| `rule` \| `llm` \| `human` |
| `ontology_version` | string | Which ontology version was used for validation |

`subject_id` and `object_id` are UUID references to the endpoint entities; they are NOT stored as properties on the Fact node but as graph edges. This enables efficient Cypher traversal and allows the same entity to appear in many relationships without duplicating the entity ID.

## Consequences

**Easier:**
- Every answer can cite `fact_id` + evidence chunk IDs.
- Temporal queries apply interval filtering before context is built.
- Candidate facts can be reviewed and promoted/rejected without losing the audit trail.
- Confidence scores allow downstream weighting in evidence fusion.

**Harder:**
- Graph writes are more complex (must write Subject→Fact→Object triple and two edges).
- Fact resolution requires traversing 2-hop paths instead of 1-hop edges.
- The PostgreSQL outbox must propagate Fact nodes to Neo4j idempotently.

**Accepted trade-offs:**
- Write amplification is acceptable because ingestion is batch and not latency-sensitive.
- Query-time graph traversal cost is bounded by the max-depth limit (3 for MVP).

## Alternatives considered

**Direct edges with properties:**
```
(A)-[:DEPENDS_ON {since: datetime, confidence: 0.9}]->(B)
```
Rejected because it mixes provenance metadata with topology and makes it impossible to attach multiple evidence chunks without bloating the edge properties.

**Separate provenance subgraph:**
```
(fact:Fact)<-[:CREATES]-(evidence:EvidenceChunk)
```
Rejected because it requires an extra JOIN on every traversal and does not cleanly separate the three knowledge layers (domain graph / subject graph / lexical graph).

**RDF-style reification:**
```
_:fact1 rdf:type :Fact
_:fact1 :subject A
_:fact1 :predicate :DEPENDS_ON
_:fact1 :object B
```
Rejected because Neo4j has no native support for RDF reification idioms and blank-node semantics complicate Cypher queries.

## References

- plan.md §3.3 (source-of-truth rule)
- plan.md §3.4 (temporal model)
- plan.md §4.2 (KnowledgeFact contract)
- ADR-002 §5.3 (outbox projection)
