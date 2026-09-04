# Agentic GraphRAG Engineering Plan

> Status: Ready for implementation  
> Primary language: Python  
> Intended executor: Coding agent working milestone by milestone  
> Initial use case: Enterprise engineering knowledge assistant for documents, source code, ADRs, tickets, deployments, and incidents  
> Last updated: 2026-09-03

## 0. Execution contract for the coding agent

This document is the authoritative implementation plan. Implement the system in the milestone order defined below. Do not silently change the architecture, replace a core technology, or skip a quality gate.

### 0.1 Working rules

1. Complete one milestone at a time.
2. Before starting a milestone, verify all of its prerequisites.
3. At the end of each milestone:
   - run its required automated tests;
   - run formatting, linting, and type checking;
   - update the progress ledger in this file;
   - record material design changes in an ADR;
   - provide a concise implementation and validation summary.
4. Do not claim completion when tests are skipped, mocked, or failing.
5. Domain logic must not depend directly on LangGraph, FastAPI, OpenAI, Neo4j, Phoenix, or another framework. Access these through application ports and infrastructure adapters.
6. Every generated answer must be traceable to evidence. Do not use a knowledge-graph relationship in an answer unless it has provenance to an authoritative structured source or source chunk.
7. Do not capture or persist hidden chain-of-thought. Persist decisions, tool inputs/outputs, retrieved evidence IDs, scores, validation results, and short machine-readable reason codes only.
8. Never log secrets, raw access tokens, credentials, or unrestricted document content.
9. All access-control filtering must happen before retrieval results enter the model context.
10. No production write tools, self-modifying prompts, automatic ontology mutation, automatic remediation, or autonomous deployment are in MVP scope.

### 0.2 Stop and request a decision only when

- credentials or access to a required external source are unavailable;
- a destructive migration would affect non-development data;
- the requested data classification conflicts with telemetry or model-provider use;
- two requirements in this document are technically incompatible;
- a core architectural replacement appears necessary rather than merely convenient.

For ordinary implementation uncertainty, choose the smallest reversible solution consistent with this document and record it in an ADR.

### 0.3 Progress ledger

Update `[ ]` to `[x]` only after the milestone acceptance criteria and validation commands pass.

- [x] M0 — Repository and engineering baseline
- [x] M1 — Local infrastructure and telemetry foundation
  - follow-up landed: Testcontainers-based component-test foundation
    (Postgres/pgvector + Neo4j fixtures, 4-layer marker split:
    unit / component / stack / fault) — see ADR-010 and
    `tests/component/`. This is M1 testing-infrastructure follow-up,
    not a new milestone.
- [ ] M2 — Domain contracts and persistence model (domain types, ports, Postgres + Neo4j adapters, Alembic migrations)
- [ ] M3 — Document ingestion and versioning
- [ ] M4 — Vector RAG baseline
- [ ] M5 — Knowledge graph construction
- [ ] M6 — Hybrid GraphRAG retrieval
- [ ] M7 — Query workflow, citations, and API
- [ ] M8 — Evaluation system and CI quality gates
- [ ] M9 — Governance, security, and adversarial testing
- [ ] M10 — Operator and review interfaces
- [ ] M11 — Production hardening and pilot readiness
- [ ] M12 — Post-MVP controlled improvement loop

---

## 1. Product definition

### 1.1 Problem

Ordinary vector RAG retrieves semantically similar text but struggles with:

- relationships across documents and systems;
- entity ambiguity and aliases;
- multi-hop dependency and impact questions;
- time-dependent facts and superseded documents;
- evidence traceability;
- reliable workflow and tool coordination;
- determining which stage caused a bad answer.

The project will build a Hybrid Agentic GraphRAG system based on two coordinated graphs:

1. **Vertical knowledge graph** — entities, facts, provenance, time, aliases, and domain constraints.
2. **Horizontal workflow graph** — explicit ingestion, retrieval, validation, and answering steps.

Each request also emits a queryable **execution graph** through structured run records and OpenTelemetry traces.

### 1.2 Primary user outcomes

The system must answer questions such as:

- Which services, APIs, queues, or database objects depend on component X?
- What could be affected if field Y or endpoint Z changes?
- Which ADR originally established this behaviour, and what later superseded it?
- Which deployment, ticket, code change, and incident are connected?
- What was true at a specified time rather than only now?
- Which exact sources support each important statement in the answer?

### 1.3 MVP success definition

MVP is complete only when all of the following are true:

- a user can ingest versioned local documents and selected repositories;
- vector-only, graph-only, and hybrid retrieval can be run and compared;
- every factual answer contains valid chunk- or structured-source citations;
- relational and multi-hop questions measurably outperform the vector-only baseline;
- every request has a correlated trace showing ingestion/retrieval/generation/validation activity;
- a versioned golden evaluation dataset runs locally and in CI;
- access-control test cases have zero unauthorized retrievals;
- unanswerable questions are refused rather than completed from model memory;
- the system can replay the recorded inputs of a failed execution against a selected configuration version.

### 1.4 Non-goals for MVP

- General web search.
- Production system mutation or remediation.
- Autonomous schema, ontology, prompt, code, or tool creation.
- Reinforcement learning or model fine-tuning.
- Uncontrolled multi-agent swarms.
- Enterprise-wide ontology coverage.
- GPU graph acceleration.
- Full RDF/OWL reasoner support.
- Mobile application.
- Kubernetes before local and single-environment deployment is proven.

---

## 2. Architectural decisions

### 2.1 Technology stack

| Concern | Decision |
|---|---|
| Core language | Python 3.12 or the newest repository-supported compatible minor version |
| Package/environment management | `uv`, with committed lock file |
| API | FastAPI |
| Data contracts | Pydantic |
| Relational access | SQLAlchemy 2.x async style |
| Database migrations | Alembic |
| Relational database | PostgreSQL |
| Vector search | pgvector |
| Knowledge graph | Neo4j using the official async Python driver and Cypher |
| Workflow runtime | LangGraph behind an internal workflow adapter |
| LLM/embedding provider | OpenAI Python SDK behind provider interfaces |
| Raw object storage | S3-compatible API; MinIO in local development |
| Cache | Redis only when measurements justify it; not required initially |
| Telemetry standard | OpenTelemetry/OTLP |
| AI trace UI | Arize Phoenix |
| Infrastructure metrics | Prometheus and Grafana |
| Tests | pytest, pytest-asyncio, Testcontainers where appropriate |
| Evaluation | DeepEval plus deterministic and custom graph evaluators |
| Experimental RAG metrics | Ragas is optional and must not be the sole release gate |
| Lint/format | Ruff |
| Static type checks | Pyright |
| Optional web UI | Next.js/TypeScript after core APIs and evals are stable |

Use the latest mutually compatible stable package versions at initialization time and commit the resolved `uv.lock`. Do not use floating dependencies in CI.

### 2.2 Framework boundary rule

The desired dependency direction is:

```text
domain <- application <- workflows/API <- infrastructure composition
```

The domain and application layers define ports such as:

- `DocumentRepository`
- `ObjectStore`
- `EmbeddingProvider`
- `VectorRetriever`
- `GraphRepository`
- `EntityExtractor`
- `EntityResolver`
- `RetrievalPlanner`
- `EvidenceReranker`
- `AnswerGenerator`
- `ClaimValidator`
- `TelemetryRecorder`
- `EvaluationRepository`

Infrastructure implements these ports. LangGraph may call application services but application services must not import LangGraph types.

### 2.3 Repository layout

```text
groundgraph/
├── apps/
│   ├── api/
│   ├── ingestion_worker/
│   ├── evaluation_runner/
│   └── web/                         # added in M10, optional before then
├── src/groundgraph/
│   ├── domain/
│   │   ├── documents/
│   │   ├── knowledge/
│   │   ├── retrieval/
│   │   ├── evidence/
│   │   ├── execution/
│   │   └── evaluation/
│   ├── application/
│   │   ├── ingestion/
│   │   ├── extraction/
│   │   ├── entity_resolution/
│   │   ├── retrieval/
│   │   ├── answering/
│   │   └── evaluation/
│   ├── workflows/
│   │   ├── ingestion_graph.py
│   │   ├── query_graph.py
│   │   └── evaluation_graph.py
│   ├── infrastructure/
│   │   ├── postgres/
│   │   ├── pgvector/
│   │   ├── neo4j/
│   │   ├── openai/
│   │   ├── object_storage/
│   │   └── telemetry/
│   └── api/
├── ontology/
│   ├── entity_types.yaml
│   ├── predicates.yaml
│   ├── constraints.yaml
│   └── versions/
├── evals/
│   ├── datasets/
│   ├── metrics/
│   ├── judges/
│   ├── regression/
│   └── reports/
├── notebooks/
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── end_to_end/
│   └── adversarial/
├── deploy/
│   ├── docker/
│   ├── prometheus/
│   ├── grafana/
│   └── phoenix/
├── docs/
│   ├── adr/
│   ├── api/
│   ├── operations/
│   └── evaluation/
├── scripts/
├── .env.example
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── uv.lock
└── plan.md
```

### 2.4 Required ADRs

Create these records as their decisions are implemented:

- ADR-001: Python and clean architecture boundaries.
- ADR-002: PostgreSQL/pgvector plus Neo4j hybrid storage.
- ADR-003: Reified facts with provenance and validity intervals.
- ADR-004: LangGraph as replaceable workflow runtime.
- ADR-005: OpenTelemetry semantic conventions and content-capture policy.
- ADR-006: Evaluation methodology and release thresholds.
- ADR-007: Pre-retrieval authorization model.
- ADR-008: Model-provider abstraction and model routing.
- ADR-009: Versioning of documents, ontology, prompts, indexes, and evaluation datasets.

---

## 3. System architecture

### 3.1 High-level flow

```text
Sources
  -> ingestion workflow
  -> raw object storage
  -> document/chunk registry in PostgreSQL
  -> embeddings in pgvector
  -> entities/facts/provenance in Neo4j

Question
  -> intent/entity analysis
  -> retrieval plan
  -> vector retrieval + graph traversal
  -> evidence fusion and reranking
  -> context sufficiency gate
  -> answer generation
  -> claim and citation validation
  -> answer or refusal

Every step
  -> OpenTelemetry traces/metrics
  -> Phoenix trace and experiment views
  -> execution/audit records in PostgreSQL
```

### 3.2 Three knowledge layers

Keep different trust levels explicit:

1. **Domain graph**
   - Entity types, predicates, constraints, permitted directions, and validation rules.
   - Stored as versioned YAML in the repository and loaded into Neo4j as metadata where useful.
2. **Subject graph**
   - Canonical entities and verified facts.
   - Facts require provenance and may have validity intervals.
3. **Lexical graph**
   - Mentions, aliases, extracted candidate relationships, text locations, and resolution candidates.
   - Candidate facts never become verified merely because an LLM produced them.

### 3.3 Source-of-truth rule for relationships

Use reified fact nodes as the authoritative representation:

```text
(subject:Entity)-[:SUBJECT_OF]->(fact:Fact)-[:OBJECT]->(object:Entity)
(fact)-[:SUPPORTED_BY]->(chunk:Chunk)
(fact)-[:DEFINED_BY]->(structured_source:SourceRecord)
```

`Fact` properties:

```text
fact_id
predicate
status = candidate | verified | rejected | superseded
confidence
valid_from
valid_to
observed_at
extraction_method = structured | rule | llm | human
extractor_version
ontology_version
created_at
updated_at
```

Direct entity-to-entity edges may later be materialized for query performance, but they are derived indexes, not authoritative facts. They must retain `fact_id` references.

### 3.4 Temporal model

Do not overwrite historical truth. Every temporal fact can contain:

- `valid_from`: when the fact became true in the domain;
- `valid_to`: when it stopped being true, null for open-ended;
- `observed_at`: when this system learned the fact;
- `superseded_by`: replacement fact or document version;
- `source_version_id`: immutable evidence version.

Queries without a supplied time use the configured current-time policy. Queries containing a historical time must apply interval filtering before context generation.

### 3.5 Identity model

All canonical identifiers are generated internally and must not expose PII:

- `source_id`
- `document_id`
- `document_version_id`
- `chunk_id`
- `entity_id`
- `fact_id`
- `ingestion_run_id`
- `execution_run_id`
- `evaluation_run_id`

External IDs are attributes scoped by source system, for example `(source_system, external_id)`. Never assume that names are unique.

---

## 4. Core domain contracts

The names below are normative even if file boundaries change slightly.

### 4.1 Document contracts

```python
class SourceDescriptor(BaseModel):
    source_id: UUID
    source_type: Literal["filesystem", "repository", "object_store", "api"]
    uri: str
    classification: str
    allowed_principals: list[str]


class ParsedDocument(BaseModel):
    document_id: UUID
    version_id: UUID
    source_id: UUID
    title: str
    media_type: str
    checksum: str
    content: str
    metadata: dict[str, JsonValue]
    effective_at: datetime | None


class Chunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    ordinal: int
    heading_path: list[str]
    content: str
    token_count: int
    checksum: str
    start_locator: str | None
    end_locator: str | None
    allowed_principals: list[str]
```

### 4.2 Knowledge contracts

```python
class EntityMention(BaseModel):
    mention_id: UUID
    chunk_id: UUID
    surface_form: str
    candidate_type: str
    locator: str | None
    extraction_confidence: float


class CanonicalEntity(BaseModel):
    entity_id: UUID
    entity_type: str
    canonical_name: str
    aliases: list[str]
    attributes: dict[str, JsonValue]


class KnowledgeFact(BaseModel):
    fact_id: UUID
    subject_id: UUID
    predicate: str
    object_id: UUID
    status: Literal["candidate", "verified", "rejected", "superseded"]
    confidence: float
    evidence_ids: list[UUID]
    valid_from: datetime | None
    valid_to: datetime | None
    observed_at: datetime
    extraction_method: Literal["structured", "rule", "llm", "human"]
    ontology_version: str
```

### 4.3 Retrieval and answer contracts

```python
class RetrievalPlan(BaseModel):
    strategy: Literal["vector", "graph", "hybrid"]
    question_type: Literal[
        "fact",
        "relationship",
        "multi_hop",
        "temporal",
        "comparison",
        "impact",
        "summary",
        "unknown",
    ]
    query_texts: list[str]
    entities: list[ResolvedEntity]
    predicates: list[str]
    max_graph_depth: int = Field(ge=0, le=3)
    vector_top_k: int = Field(ge=1, le=50)
    final_evidence_limit: int = Field(ge=1, le=30)
    valid_at: datetime | None
    reason_codes: list[str]


class Evidence(BaseModel):
    evidence_id: UUID
    source_id: UUID
    document_id: UUID | None
    chunk_id: UUID | None
    structured_record_id: str | None
    content: str
    retrieval_method: Literal["vector", "keyword", "graph", "structured"]
    vector_score: float | None
    rerank_score: float | None
    graph_path_fact_ids: list[UUID]
    valid_from: datetime | None
    valid_to: datetime | None
    allowed_principals: list[str]


class AnswerClaim(BaseModel):
    claim_id: UUID
    text: str
    factual: bool
    evidence_ids: list[UUID]
    support_status: Literal["supported", "partially_supported", "unsupported"]


class QueryResponse(BaseModel):
    execution_run_id: UUID
    answer: str | None
    status: Literal["answered", "clarification_required", "insufficient_evidence", "failed"]
    claims: list[AnswerClaim]
    citations: list[Citation]
    confidence_band: Literal["high", "medium", "low"]
    warnings: list[str]
```

### 4.4 Structured-output policy

- All LLM steps return validated Pydantic structures.
- Invalid model output receives at most one format-repair retry.
- A second failure terminates that node with a typed error.
- Do not parse model output using ad hoc regular expressions when structured output is available.
- Store prompt template ID, prompt version, model alias, provider model ID, and schema version with every model span.

---

## 5. Storage model

### 5.1 PostgreSQL tables

Implement migrations for at least:

- `sources`
- `source_sync_state`
- `documents`
- `document_versions`
- `chunks`
- `chunk_embeddings`
- `ingestion_runs`
- `ingestion_steps`
- `execution_runs`
- `execution_steps`
- `retrieval_results`
- `answer_claims`
- `citations`
- `user_feedback`
- `prompt_versions`
- `model_config_versions`
- `index_versions`
- `evaluation_datasets`
- `evaluation_cases`
- `evaluation_runs`
- `evaluation_results`
- `human_review_items`

Requirements:

- use UUID primary keys;
- use UTC timestamps;
- retain immutable document versions;
- store checksums to support idempotency;
- use explicit foreign keys and deletion behaviour;
- include tenant/security scope on all retrievable content;
- index foreign keys, checksums, statuses, validity fields, and execution correlation IDs;
- pgvector dimensionality must come from the configured embedding model and be migration-controlled;
- never mix embeddings from incompatible embedding model/version configurations in the same searchable index partition.

### 5.2 Neo4j constraints and indexes

Create uniqueness constraints for:

- `Entity.entity_id`
- `Fact.fact_id`
- `Document.document_id`
- `DocumentVersion.version_id`
- `Chunk.chunk_id`
- `OntologyType.type_id`
- `OntologyPredicate.predicate_id`

Create indexes for:

- entity canonical name and normalized aliases;
- entity type;
- fact predicate and status;
- fact validity intervals;
- chunk/document/source references.

All Cypher writes must be idempotent and transactionally scoped per ingestion unit.

### 5.3 Consistency between PostgreSQL and Neo4j

PostgreSQL is the source of truth for ingestion status, source documents, chunks, and run state. Neo4j is the source of truth for graph entities and facts.

Use an outbox-based projection flow:

1. Store parsed document/chunks and an outbox event in one PostgreSQL transaction.
2. Worker consumes event and writes graph projection idempotently.
3. Mark the outbox event complete only after Neo4j succeeds.
4. Reconciliation job compares expected document versions with graph projections.

Do not attempt an unsafe distributed transaction between PostgreSQL and Neo4j.

---

## 6. Ingestion design

### 6.1 Initial source formats

MVP must support:

- Markdown
- plain text
- HTML
- PDF with text layer
- DOCX
- EPUB
- selected source-code repositories

Scanned/OCR-only PDFs may be rejected with an explicit unsupported reason in MVP. OCR can be added later.

### 6.2 Ingestion workflow

```text
register_source
-> acquire_source
-> calculate_checksum
-> detect_change
-> persist_raw_file
-> parse_document
-> normalize_content
-> classify_document
-> create_semantic_chunks
-> persist_document_and_chunks
-> generate_embeddings
-> extract_mentions_and_candidate_facts
-> resolve_entities
-> validate_facts_against_ontology
-> project_to_graph
-> run_ingestion_quality_checks
-> publish_index_version
```

### 6.3 Idempotency and versioning

- Same source plus same checksum must not create a new version.
- Changed content creates an immutable document version.
- Removing a source marks its current version inactive and removes it from active retrieval, while preserving audit history.
- Reprocessing with a new parser, chunker, embedding model, extraction prompt, or ontology version creates a distinguishable index version.
- Failed ingestion can resume from the last durable completed step.

### 6.4 Chunking

Implement format-aware, structure-aware chunking:

- preserve heading paths;
- do not split code blocks, tables, or list items arbitrarily;
- maintain parent document/version IDs;
- include local locators for citations;
- allow limited overlap but measure duplicate-context rate;
- make target size and overlap configurable;
- record `chunker_version` and configuration hash.

Start with sensible defaults, then tune using retrieval evaluation. Do not assume one chunk size is optimal for all formats.

### 6.5 Code ingestion

For repositories, start with:

- repository, directory, file, module, symbol, function/class, import, and call relationships;
- README, ADR, API/schema, configuration, and deployment files;
- Tree-sitter or language-native parsers where available;
- file path, commit SHA, language, and symbol locator as metadata.

Do not attempt full semantic compilation for every language in MVP. Support Python and TypeScript first, then add languages through adapters.

### 6.6 Ingestion quality checks

At minimum calculate:

- parse success rate;
- extracted text size versus source size sanity check;
- empty and near-empty chunk rate;
- duplicate chunk rate;
- missing metadata rate;
- embedding coverage;
- entity extraction rate;
- unresolved mention rate;
- candidate/verified/rejected fact counts;
- provenance coverage;
- ingestion latency and cost.

An index version cannot be published when required-source parsing, embedding coverage, authorization metadata, or provenance validation fails.

---

## 7. Knowledge graph and ontology

### 7.1 Initial entity types

Keep the first ontology deliberately small:

- `Repository`
- `Module`
- `Service`
- `API`
- `Database`
- `Table`
- `Column`
- `QueueOrTopic`
- `BusinessProcess`
- `Requirement`
- `Ticket`
- `ADR`
- `Deployment`
- `Incident`
- `Team`
- `PersonRole` — not raw personal profile data
- `Document`
- `Chunk`

### 7.2 Initial predicates

- `CONTAINS`
- `DEPENDS_ON`
- `CALLS`
- `EXPOSES`
- `READS_FROM`
- `WRITES_TO`
- `PUBLISHES_TO`
- `CONSUMES_FROM`
- `IMPLEMENTS`
- `CHANGES`
- `GOVERNS`
- `SUPERSEDES`
- `DEPLOYED_AS`
- `AFFECTED`
- `OWNED_BY`
- `MENTIONS`
- `SUPPORTED_BY`

Each predicate definition must specify:

- allowed subject types;
- allowed object types;
- inverse name where applicable;
- whether it is directional;
- whether multiple concurrent facts are valid;
- whether temporal validity is required;
- minimum evidence policy;
- default trust based on extraction method.

### 7.3 Extraction pipeline

Use this order:

1. Deterministic extraction from structured sources.
2. Rule-based extraction from code/configuration.
3. LLM structured extraction from unstructured text.
4. Entity resolution.
5. Ontology validation.
6. Confidence and evidence checks.
7. Candidate or verified status assignment.

LLM-extracted facts default to `candidate`. Automatic verification is allowed only when an explicit predicate policy permits it and evidence rules pass.

### 7.4 Entity resolution pipeline

```text
normalize mention
-> candidate generation by type/name/alias/external ID
-> attribute and neighborhood comparison
-> evidence-based score
-> auto-link, create-new, or human-review decision
```

Initial decision bands should be configurable, for example:

- high confidence: automatic link;
- medium confidence: human review;
- low confidence: create candidate entity or leave unresolved.

Do not hardcode confidence thresholds inside resolver logic. Store decision features and reason codes for evaluation.

### 7.5 Graph retrieval safety

- Graph queries are generated from an allowlisted predicate vocabulary.
- Maximum traversal depth is 3 for MVP.
- Apply tenant and principal filters at seed selection and traversal.
- Apply validity-time filters inside the Cypher query.
- Limit maximum expanded nodes, paths, and execution time.
- Return fact IDs and provenance with every selected path.
- Reject paths containing candidate/rejected/superseded facts unless the query explicitly requests historical or uncertain information.

---

## 8. Retrieval and answering

### 8.1 Retrieval strategies

Implement three independently callable strategies:

1. **Vector** — semantic plus keyword retrieval and reranking.
2. **Graph** — entity seed resolution, constrained traversal, then provenance retrieval.
3. **Hybrid** — run vector and graph retrieval, normalize scores, deduplicate, fuse, and rerank.

Always retain strategy-specific results so evaluation can compare them.

### 8.2 Query workflow

```text
authorize_request
-> classify_question
-> extract_query_entities
-> resolve_query_entities
-> build_retrieval_plan
-> retrieve_vector_evidence
-> retrieve_graph_paths
-> hydrate_graph_provenance
-> fuse_and_rerank_evidence
-> evaluate_context_sufficiency
-> optionally_revise_plan_and_retry_once
-> generate_structured_answer
-> extract_atomic_claims
-> validate_claim_support
-> validate_citations
-> policy_gate
-> return_answer_or_refusal
-> persist_execution_summary
```

### 8.3 Routing policy

Start with explainable rules plus structured LLM classification:

| Question | Default strategy |
|---|---|
| Single-source fact | Vector |
| Summary of a known document | Vector |
| Entity relationship | Hybrid |
| Multi-hop dependency | Hybrid |
| Impact analysis | Hybrid |
| Historical state | Hybrid with temporal filters |
| Explicit graph inspection | Graph |
| Unknown/ambiguous entity | Clarify or Hybrid with strict confidence gate |

Record the chosen strategy and reason codes. The evaluator must be able to override the strategy to compare candidates.

### 8.4 Retrieval budgets

Initial configurable limits:

- vector retrieval candidates: 30;
- keyword retrieval candidates: 20;
- reranked final vector evidence: 10;
- graph depth: 2 default, 3 maximum;
- graph candidate paths: 20;
- final combined evidence items: 12;
- retrieval retries: 1;
- format repair retries per LLM node: 1;
- request wall-clock timeout: configurable by environment.

Tune using evaluation data rather than intuition.

### 8.5 Evidence fusion

Fusion must:

- deduplicate exact and near-duplicate chunks;
- retain the strongest provenance record;
- normalize scores across retrieval methods;
- reward corroboration from independent sources;
- penalize superseded or temporally invalid evidence;
- prevent a high-degree graph hub from dominating context;
- keep contradictory evidence visible to the context evaluator;
- preserve source diversity where useful.

### 8.6 Context sufficiency gate

Before answer generation, determine:

- whether the requested entities were resolved confidently;
- whether required relationship paths were found;
- whether relevant facts have evidence;
- whether evidence is current for the requested time;
- whether sources conflict;
- whether the user has permission to access enough evidence;
- whether another bounded retrieval attempt is likely to help.

Possible results:

- `sufficient`
- `retry_with_revised_plan`
- `clarification_required`
- `insufficient_evidence`
- `conflicting_evidence`

### 8.7 Claim and citation validation

After generating a draft:

1. Split the answer into atomic claims.
2. Mark claims factual or non-factual.
3. Map every factual claim to one or more evidence IDs.
4. Verify that cited evidence actually supports the claim.
5. Remove or revise unsupported claims once.
6. If unsupported factual claims remain, return a refusal/partial-answer status rather than an apparently complete answer.

Citation locators must include document title/version and the most precise available location: heading, page, paragraph, line, symbol, or chunk locator.

### 8.8 Answering policy

- Use only supplied evidence for factual claims.
- Clearly distinguish verified fact, inference, uncertainty, and conflict.
- Do not fill knowledge gaps from model memory.
- Prefer a short, supported answer over a comprehensive unsupported answer.
- For impact analysis, describe found paths and note graph coverage limitations.
- For historical queries, state the effective time used.

---

## 9. API specification

### 9.1 Required endpoints

```text
POST   /v1/sources
GET    /v1/sources/{source_id}
POST   /v1/sources/{source_id}/ingestions
GET    /v1/ingestions/{ingestion_run_id}
POST   /v1/query
GET    /v1/executions/{execution_run_id}
GET    /v1/executions/{execution_run_id}/evidence
POST   /v1/executions/{execution_run_id}/feedback
POST   /v1/evaluations
GET    /v1/evaluations/{evaluation_run_id}
GET    /v1/reviews
POST   /v1/reviews/{review_id}/decision
GET    /health/live
GET    /health/ready
```

### 9.2 Query request

```json
{
  "question": "Which services would be affected if API X changes?",
  "strategy": "auto",
  "valid_at": null,
  "conversation_id": null,
  "filters": {
    "source_ids": [],
    "document_types": []
  },
  "debug": false
}
```

Authorization identity and principal scopes come from authenticated request context, never from client-supplied body fields.

### 9.3 Error model

Use stable problem-detail responses with:

- code;
- message safe for the caller;
- correlation/trace ID;
- retryability;
- validation details where safe.

Do not expose prompts, credentials, internal stack traces, raw Cypher, or unauthorized resource identifiers.

---

## 10. Observability design

Observability is mandatory from M1 onward. No application workflow node is complete until it emits the required trace data and metrics.

### 10.1 Trace hierarchy

One user query should resemble:

```text
rag.request
├── auth.scope
├── question.classify
├── entity.extract
├── entity.resolve
├── retrieval.plan
├── retrieval.vector
│   ├── embedding.create
│   ├── pgvector.search
│   ├── keyword.search
│   └── evidence.rerank
├── retrieval.graph
│   ├── graph.seed_entities
│   ├── graph.traverse
│   └── graph.hydrate_provenance
├── evidence.fuse
├── context.evaluate
├── answer.generate
├── claims.extract
├── claims.validate
├── citations.validate
├── policy.gate
└── execution.persist
```

Ingestion and evaluation runs require equivalent root traces: `rag.ingestion` and `rag.evaluation`.

### 10.2 Required span attributes

At request level:

- `service.name`
- `deployment.environment`
- `tenant.id_hash`
- `conversation.id`
- `execution.run_id`
- `workflow.name`
- `workflow.version`
- `ontology.version`
- `index.version`
- `prompt.bundle_version`
- `retrieval.strategy`
- `question.type`
- `request.status`

At model-call level:

- operation name;
- provider and model ID;
- prompt template/version, not necessarily raw prompt;
- structured-output schema/version;
- input/output token counts;
- latency;
- retry count;
- finish reason;
- estimated cost;
- error type.

At vector retrieval level:

- query hash and safe normalized query metadata;
- filters applied;
- top-K requested/returned;
- candidate chunk IDs;
- similarity and reranking score summaries;
- embedding/index version;
- duration.

At graph retrieval level:

- seed entity IDs and resolution-confidence bands;
- allowed predicates;
- requested/actual depth;
- candidate and selected path counts;
- selected fact IDs;
- rejected-path reason counts;
- graph/ontology version;
- duration and timeout state.

At validation level:

- factual claim count;
- supported/partial/unsupported counts;
- citation coverage;
- citation correctness when evaluated;
- policy decision;
- refusal reason.

### 10.3 Content capture and privacy

Default production policy:

- do not export raw prompts, responses, chunks, or personal identifiers to general telemetry;
- store hashes, IDs, sizes, scores, versions, and reason codes;
- permit sampled content capture only in explicitly approved, access-controlled environments;
- redact secrets and configured PII patterns before any content leaves the application;
- use separate retention policies for operational traces and approved evaluation examples;
- never record hidden reasoning content.

### 10.4 Application metrics

Operational:

- request rate, error rate, retry rate;
- P50/P95/P99 end-to-end latency;
- parse, embedding, vector, graph, reranking, generation, and validation latency;
- worker queue depth and age;
- PostgreSQL/Neo4j/object-store dependency health;
- token usage and estimated cost by workflow/model/tenant.

RAG quality proxies:

- empty retrieval rate;
- low-score retrieval rate;
- second-retrieval rate;
- citation coverage rate;
- unsupported-claim rate;
- refusal and clarification rate;
- duplicate-context rate;
- stale-evidence rate;
- negative feedback and correction rate.

Graph quality:

- entity resolution success and review rates;
- duplicate-entity and orphan-node counts;
- candidate/verified/rejected fact counts;
- fact provenance coverage;
- invalid and timed-out path rates;
- stale/temporally conflicting fact counts;
- graph projection lag;
- PostgreSQL-to-Neo4j reconciliation failures.

### 10.5 Required dashboards

1. **Operations** — traffic, failures, latency, dependencies, queues, resource use.
2. **Retrieval** — strategies, scores, top-K, retries, vector versus graph contribution.
3. **Answer quality** — citations, claims, refusals, feedback, evaluation trends.
4. **Knowledge graph health** — nodes/facts, provenance, duplicates, resolution, staleness, projection lag.
5. **Cost** — tokens and estimated cost per request, stage, model, tenant, and configuration version.

### 10.6 Initial SLOs and alerts

Initial pilot targets, to be revised from measurements:

- availability: at least 99.5% during pilot service hours;
- P95 answer latency: at most 8 seconds for non-streaming standard queries;
- P95 vector retrieval: at most 500 ms on pilot dataset;
- P95 graph traversal: at most 1 second on pilot dataset;
- unauthorized retrieval: exactly 0;
- published-index ingestion lag: at most 1 hour for scheduled sources;
- citation coverage: at least 95% on evaluated factual claims;
- unsupported factual claim rate: below 3% on production samples.

Alert on:

- any authorization leakage;
- unsupported-claim or citation failures above threshold;
- sudden empty-retrieval increase;
- graph projection/reconciliation failure;
- ingestion lag breach;
- P95 latency breach over a sustained window;
- cost per request increase greater than 25% against the current baseline;
- entity resolution review/failure spike.

---

## 11. Evaluation system

### 11.1 Evaluation principles

1. Evaluate ingestion, retrieval, graph construction, answers, and workflow trajectories separately.
2. Prefer deterministic metrics over LLM judges where deterministic truth exists.
3. Never allow a single aggregate score to hide a security, provenance, or citation failure.
4. Compare every material change with a versioned baseline.
5. Store dataset, prompt, ontology, index, model, workflow, and code version with each result.
6. Human-labelled cases are the release authority; synthetic cases expand coverage but do not replace expert review.

### 11.2 Evaluation dataset schema

Store golden cases in JSONL and import them into versioned database datasets:

```json
{
  "case_id": "impact-001",
  "question": "Which services depend on API X?",
  "question_type": "impact",
  "answerable": true,
  "expected_answer": "A concise reference answer",
  "required_evidence_ids": ["doc-v1#chunk-12"],
  "acceptable_evidence_ids": ["adr-4#decision"],
  "expected_entity_ids": ["entity-api-x"],
  "expected_fact_paths": [
    ["fact-service-a-calls-api-x"],
    ["fact-service-b-depends-service-a", "fact-service-a-calls-api-x"]
  ],
  "valid_at": "2026-09-01T00:00:00Z",
  "principal_scope": ["engineering"],
  "tags": ["multi_hop", "graph", "temporal"],
  "human_notes": "Why this answer and evidence are correct"
}
```

### 11.3 Dataset composition

Before production pilot, build:

- at least 100 human-authored golden cases;
- 300–500 synthetic expansion cases reviewed by sampling;
- at least 50 genuinely unanswerable cases;
- at least 50 ambiguous/conflicting/temporal cases;
- at least 50 adversarial and prompt-injection cases;
- at least 30 access-control cases spanning allowed and forbidden principals.

Golden cases must cover:

- single-hop fact retrieval;
- document summary;
- entity aliases and ambiguity;
- relationships and multi-hop paths;
- impact analysis;
- historical state;
- superseded evidence;
- conflicting sources;
- insufficient evidence;
- unauthorized evidence;
- malformed input and injection attempts.

### 11.4 Deterministic ingestion metrics

- parse success and expected section recovery;
- chunk-boundary fixture assertions;
- metadata completeness;
- checksum/idempotency behaviour;
- embedding coverage;
- deletion/deactivation correctness;
- graph projection completeness;
- provenance coverage;
- index-version consistency.

### 11.5 Vector retrieval metrics

- Recall@K;
- Precision@K;
- Mean Reciprocal Rank;
- NDCG@K;
- evidence coverage;
- source diversity;
- reranker lift over pre-reranked order;
- latency and cost.

### 11.6 Graph retrieval metrics

Implement custom evaluators for:

- entity-linking accuracy;
- entity-resolution precision/recall on labelled pairs;
- expected seed-entity recall;
- predicate-selection accuracy;
- graph path precision;
- graph path recall;
- path validity against ontology;
- provenance coverage;
- temporal validity;
- supporting-evidence coverage;
- traversal efficiency and irrelevant-node expansion;
- authorization-path leakage.

### 11.7 Answer metrics

Use deterministic checks and DeepEval/custom judges for:

- answer correctness;
- answer relevance;
- faithfulness/groundedness;
- claim-level evidence support;
- citation coverage;
- citation correctness;
- structured-output validity;
- appropriate abstention;
- contradiction handling;
- instruction following.

Citation formulas:

```text
citation_coverage = supported factual claims / all factual claims
citation_correctness = citations that support their claims / all citations
```

Security and authorization metrics are binary gates, not weighted quality metrics.

### 11.8 Workflow/trajectory metrics

- correct retrieval-strategy selection;
- correct tool/node sequence;
- tool argument validity;
- unnecessary tool-call count;
- retry-budget compliance;
- graph-depth and evidence-budget compliance;
- context gate correctness;
- validation node never skipped;
- task completion;
- deterministic replay compatibility.

### 11.9 LLM judge calibration

Before using an LLM judge as a release gate:

1. Human-label at least 100 representative outputs.
2. Run the judge on the same outputs without access to human labels.
3. Compare agreement, false positives, false negatives, and score distributions.
4. Refine the rubric and examples.
5. Record judge model, prompt, rubric, and calibration dataset versions.
6. Require acceptable agreement chosen in ADR-006; start with a target Cohen's kappa of at least 0.75.
7. Recalibrate after changing judge model or rubric and periodically on production samples.

Do not let the answer-generating model be the only judge of its own output.

### 11.10 Evaluation tiers

**Local developer loop**

- unit and component tests;
- 10–20 fast evaluation cases;
- no network-dependent judge unless explicitly selected.

**Pull request gate**

- full deterministic tests;
- 30–50 representative golden cases;
- retrieval, graph, citation, and security thresholds;
- candidate-versus-baseline comparison;
- publish a machine-readable and Markdown report artifact.

**Nightly evaluation**

- full golden and adversarial datasets;
- vector-only versus graph-only versus hybrid comparison;
- model/prompt/index configuration experiments;
- cost and latency analysis;
- drift and graph consistency checks.

**Pre-release evaluation**

- full dataset plus human review sample;
- canary/shadow comparison against current release;
- security review and rollback verification.

### 11.11 Initial quality gates

These are pilot thresholds, revised only through ADR and evidence:

| Metric | Gate |
|---|---:|
| Parsing success on supported fixtures | 100% |
| Required metadata and ACL coverage | 100% |
| Retrieval Recall@10 | >= 0.85 |
| Entity-linking accuracy | >= 0.95 |
| Verified-fact precision | >= 0.95 |
| Graph path recall on graph-labelled cases | >= 0.85 |
| Graph path validity | >= 0.98 |
| Fact provenance coverage | 100% |
| Citation coverage | >= 0.95 |
| Citation correctness | >= 0.95 |
| Faithfulness | >= 0.90 |
| Appropriate abstention | >= 0.85 |
| Structured output success after allowed retry | 100% |
| Unauthorized retrieval | 0 |
| Regression against accepted baseline | no critical regression; <= 2% on agreed aggregate metrics |

No aggregate improvement may override unauthorized retrieval, missing provenance, or a critical citation regression.

---

## 12. Security and governance

### 12.1 Authorization

- Authenticate requests through a replaceable identity adapter.
- Convert identity into server-controlled principal and tenant scopes.
- Propagate scopes through query planning.
- Apply authorization filters in PostgreSQL, pgvector, and Neo4j before returning candidates.
- Recheck evidence authorization before context construction.
- Never trust caller-supplied principals.
- Include denied-result counts, not denied identities/content, in safe telemetry.

### 12.2 Data classification and retention

Every source/document has:

- classification;
- owner;
- allowed principals;
- retention class;
- telemetry-content policy;
- model-provider eligibility.

Deletion must remove active searchability from raw storage, PostgreSQL/pgvector, Neo4j projections, caches, and future model contexts while preserving only permitted audit metadata.

### 12.3 Prompt injection defenses

- Treat retrieved content as untrusted data, never executable instructions.
- Delimit evidence from system/developer instructions.
- Detect common injection indicators and assign a risk score.
- Do not let retrieved content alter tool permissions, policies, model configuration, or system prompts.
- Allowlist workflow nodes and tools.
- Validate structured output and all tool arguments.
- Include malicious-document cases in adversarial evaluation.

### 12.4 Secrets

- `.env.example` contains names and safe examples only.
- Local secrets remain uncommitted.
- Production secrets use the platform secret manager.
- Add secret scanning to CI.
- Rotate any credential that is accidentally logged or committed; do not merely delete the line.

### 12.5 Auditability

Retain:

- authenticated actor hash and tenant;
- request and execution IDs;
- workflow, prompt, ontology, model, and index versions;
- retrieval strategy and evidence IDs;
- selected graph fact paths;
- tool call metadata;
- policy and validation decisions;
- feedback/review decisions;
- deployment version.

Do not retain hidden reasoning or prohibited document content.

---

## 13. Milestone implementation plan

## M0 — Repository and engineering baseline

### Objective

Create a reproducible Python repository with strict quality checks and documented commands.

### Tasks

- Initialize the project and `uv` environment.
- Configure `src` layout, Ruff, Pyright, pytest, and coverage.
- Create application settings using Pydantic Settings.
- Add `.env.example`, `.gitignore`, editor configuration, and pre-commit hooks if used.
- Add Makefile or equivalent task commands:
  - `make setup`
  - `make format`
  - `make lint`
  - `make typecheck`
  - `make test`
  - `make test-integration`
  - `make eval-smoke`
  - `make check`
- Add base exception hierarchy and result/error conventions.
- Add CI workflow running install, lint, type check, unit tests, and dependency/secret scans.
- Add ADR template and create ADR-001.

### Required tests

- settings load with valid environment;
- startup fails clearly for missing required configuration;
- one domain model validation test;
- one async service test;
- CI command executes locally.

### Acceptance criteria

- clean checkout can be set up from README;
- `uv.lock` is committed;
- `make check` passes;
- no application code imports undeclared packages;
- test, lint, and type failures cause nonzero exit status.

---

## M1 — Local infrastructure and telemetry foundation

### Objective

Provide reproducible local services and trace a minimal API request end to end before RAG logic is built.

### Tasks

- Add Docker Compose services for PostgreSQL/pgvector, Neo4j, MinIO, Phoenix, OpenTelemetry Collector, Prometheus, and Grafana.
- Add health checks and persistent development volumes.
- Create FastAPI service with live/ready endpoints.
- Configure structured JSON logging and trace/log correlation IDs.
- Configure OpenTelemetry SDK and OTLP export through the collector.
- Send a root API span and dependency spans to Phoenix.
- Expose Prometheus metrics.
- Add local Grafana provisioning and a starter operations dashboard.
- Define telemetry redaction/content-capture policy.
- Create ADR-002 and ADR-005.

### Required tests

- Compose configuration validation;
- health/readiness behaviour with healthy and unavailable dependencies;
- trace propagation through one API request and one background task;
- secrets and raw authorization headers are absent from captured telemetry;
- Prometheus scrape endpoint works.

### Acceptance criteria

- one command starts all local dependencies;
- Phoenix displays a correlated request trace;
- Grafana displays request count and latency;
- readiness reports the correct failed dependency;
- telemetry does not contain raw secret values.

---

## M2 — Domain contracts and persistence model

### Objective

Implement stable domain types, ports, PostgreSQL schema, Neo4j constraints, and execution records.

### Tasks

- Implement the contracts in Section 4.
- Define repository and provider protocols/interfaces.
- Add SQLAlchemy models and Alembic migrations from Section 5.1.
- Enable pgvector and add embedding/index version fields.
- Add Neo4j schema creation/migration utility.
- Implement execution run/step state machine.
- Add transactional outbox tables and worker contract.
- Add prompt/model/index configuration version records.
- Add repository integration adapters.
- Create ADR-003 and ADR-009.

### Required tests

- Pydantic validation and serialization round trips;
- database migration up on an empty database;
- repository CRUD and transaction rollback;
- uniqueness/idempotency constraints;
- outbox claim/retry semantics;
- Neo4j constraints and simple fact round trip;
- execution step state transitions reject invalid transitions.

### Acceptance criteria

- schemas can be created from scratch through migrations only;
- all repository integration tests run against real disposable services;
- no infrastructure type leaks into domain contracts;
- a sample execution DAG can be persisted and reconstructed.

---

## M3 — Document ingestion and versioning

### Objective

Ingest supported formats into immutable documents and semantic chunks with provenance, ACLs, idempotency, and telemetry.

### Tasks

- Implement filesystem/object upload source registration.
- Implement safe file acquisition and checksum calculation.
- Store raw source bytes in object storage.
- Implement parsers for Markdown, text, HTML, PDF, DOCX, and EPUB.
- Implement code repository scanner for Python and TypeScript.
- Implement structure-aware chunker and locator generation.
- Persist document versions and chunks.
- Implement source deletion/deactivation behaviour.
- Add ingestion workflow using ordinary application services, then compose with LangGraph adapter if beneficial.
- Emit ingestion spans and metrics.
- Build format fixtures, including malformed and empty files.

### Required tests

- golden parser tests for every supported format;
- heading/table/code-block preservation fixtures;
- duplicate ingestion is idempotent;
- changed content creates a new immutable version;
- source deactivation removes active retrievability;
- ACL metadata reaches every chunk;
- failure resumes without duplicating completed data;
- parser and chunker telemetry contains source/version IDs but no prohibited content.

### Acceptance criteria

- all supported fixtures ingest successfully;
- unsupported/scanned inputs fail with typed reason codes;
- published chunks retain precise locators and source versions;
- required metadata/ACL coverage is 100%;
- ingestion quality report is generated per run.

---

## M4 — Vector RAG baseline

### Objective

Create a measurable vector-plus-keyword RAG baseline before adding graph retrieval.

### Tasks

- Implement embedding provider port and OpenAI adapter.
- Batch embeddings with rate-limit, timeout, and retry handling.
- Store embedding model/version/dimensions and reject incompatible mixing.
- Implement pgvector similarity search with pre-retrieval ACL and metadata filters.
- Implement PostgreSQL full-text/keyword retrieval.
- Implement reciprocal-rank or another documented fusion baseline.
- Implement reranker port and first adapter.
- Create baseline answer generator using evidence-only prompt and structured output.
- Add simple citation creation from chunk locators.
- Add trace spans and metrics for embedding, retrieval, reranking, and generation.
- Create the first 30–50 human golden cases.

### Required tests

- embedding batching and partial retry;
- dimension/config mismatch rejection;
- ACL filtering before retrieval results are returned;
- metadata and version filters;
- deterministic fusion tests;
- citation points to the exact chunk/version;
- unanswerable fixture produces refusal;
- vector baseline evaluation report.

### Acceptance criteria

- vector-only query API works end to end;
- baseline Recall@10 and answer metrics are recorded, not necessarily final-gate compliant yet;
- every generated factual claim has a candidate citation;
- trace explains candidate selection, reranking, token use, latency, and cost;
- baseline result is versioned for later GraphRAG comparison.

---

## M5 — Knowledge graph construction

### Objective

Build a provenance-aware, temporal knowledge graph from structured, code, and unstructured sources.

### Tasks

- Create versioned ontology YAML and validation loader.
- Implement deterministic extractors for repository/code/config relationships.
- Implement LLM structured entity/fact extraction for prose.
- Persist mentions and candidate resolution data.
- Implement candidate generation, scoring, decision bands, and review queue.
- Implement ontology constraint validator.
- Implement reified fact writes with evidence and validity.
- Implement outbox graph projection and reconciliation job.
- Implement fact supersession and source-version deactivation.
- Add graph health metrics and dashboard.
- Create labelled entity-resolution and fact-extraction evaluation fixtures.

### Required tests

- ontology schema and constraint tests;
- deterministic extractor golden tests;
- malformed LLM output and retry behaviour;
- candidate facts cannot bypass verification policy;
- fact without provenance is rejected;
- entity alias, collision, and ambiguity cases;
- graph writes are idempotent;
- temporal supersession preserves history;
- reconciliation detects missing/extra projections;
- ACL attributes exist on graph-retrievable objects.

### Acceptance criteria

- initial entity and predicate ontology is queryable;
- verified-fact precision reaches the initial gate on labelled fixtures;
- provenance coverage for searchable facts is 100%;
- ambiguous entities enter human review rather than being silently merged;
- graph health dashboard shows projection, resolution, provenance, and conflict metrics.

---

## M6 — Hybrid GraphRAG retrieval

### Objective

Add safe graph traversal and hybrid evidence fusion, and prove the improvement against vector-only retrieval.

### Tasks

- Implement query-time entity extraction and resolution.
- Implement `RetrievalPlan` generator with explainable reason codes.
- Implement allowlisted, parameterized Cypher traversal.
- Apply temporal, status, tenant, and principal filters inside graph queries.
- Hydrate selected fact paths with their supporting chunks/structured sources.
- Implement vector, graph, and hybrid strategies behind one interface.
- Implement score normalization, deduplication, corroboration, staleness penalty, and source diversity.
- Implement path and evidence budgets.
- Add graph path visualization payload to debug response, excluding unauthorized details.
- Add custom GraphRAG evaluators from Section 11.6.
- Expand golden set with relationship, multi-hop, temporal, and impact questions.

### Required tests

- expected entity seeds and fact paths;
- candidate/rejected/superseded facts excluded correctly;
- historical `valid_at` query returns historical rather than current state;
- maximum graph depth and expansion budget enforced;
- Cypher injection attempts fail safely;
- unauthorized nodes/paths cannot appear in candidates or telemetry;
- hybrid fusion deterministic fixtures;
- vector-only, graph-only, and hybrid experiment comparison.

### Acceptance criteria

- all three strategies are independently executable;
- graph path validity and provenance gates pass;
- hybrid improves multi-hop/relationship accuracy by a target of at least 15% relative to the accepted vector baseline, or an ADR documents evidence-based remediation before proceeding;
- no ACL leakage occurs;
- retrieval traces expose safe path/fact IDs, scores, filters, and timing.

---

## M7 — Query workflow, citations, and API

### Objective

Compose the complete bounded workflow with context evaluation, retry, claim validation, refusal, and stable APIs.

### Tasks

- Implement all query nodes listed in Section 8.2 as application services.
- Compose nodes through the LangGraph adapter.
- Add durable checkpoints where useful without coupling domain state to LangGraph.
- Implement one bounded retrieval retry.
- Implement atomic claim extraction.
- Implement deterministic citation coverage plus claim-support evaluator.
- Add policy gate and typed answer/refusal statuses.
- Implement required API endpoints and OpenAPI documentation.
- Persist execution summaries and evidence references.
- Implement replay from recorded safe inputs/configuration IDs.
- Add contract tests for all endpoints.
- Create ADR-004 and ADR-008.

### Required tests

- every workflow branch, including retry, clarify, refuse, conflict, and failure;
- node state schema validation;
- retry and timeout budgets;
- validation cannot be bypassed;
- unsupported claims are removed or cause partial/refusal status;
- citation locators resolve to stored versions;
- replay uses requested versions and does not mutate original run;
- API authorization and problem-detail errors;
- trace hierarchy and correlation IDs.

### Acceptance criteria

- end-to-end workflow passes representative cases;
- no factual answer is returned with unresolved unsupported claims;
- execution can be inspected and replayed;
- API contract is documented and stable;
- request traces show every node and all safe diagnostic metadata.

---

## M8 — Evaluation system and CI quality gates

### Objective

Make quality measurable, reproducible, and enforceable in development and deployment.

### Tasks

- Implement versioned JSONL dataset loader and database import.
- Implement deterministic ingestion, retrieval, citation, graph, temporal, and security evaluators.
- Integrate DeepEval behind an evaluation adapter.
- Add optional Ragas experiment adapter without making it a release dependency.
- Implement judge prompts/rubrics as versioned assets.
- Build judge calibration notebook/script and report.
- Implement candidate-versus-baseline experiment runner.
- Export results to PostgreSQL, Phoenix experiments, JSON, and Markdown.
- Add PR smoke gate and nightly full evaluation workflows.
- Add regression classification: critical, major, minor, informational.
- Add evaluation trend dashboard.
- Complete at least 100 human golden cases and required special datasets.
- Create ADR-006.

### Required tests

- metric formula unit tests;
- dataset schema and duplicate-case validation;
- deterministic reproduction with fixed fixtures;
- evaluator timeout/error isolation;
- baseline comparison and threshold failure behaviour;
- judge output schema and calibration calculations;
- report generation;
- CI fails on a deliberately injected retrieval/citation/security regression.

### Acceptance criteria

- `make eval-smoke` is reliable enough for local and PR use;
- nightly evaluation covers full datasets;
- results identify which component regressed rather than only giving one total score;
- security, provenance, and citation critical gates cannot be overridden by aggregate score;
- accepted baseline is versioned and reproducible.

---

## M9 — Governance, security, and adversarial testing

### Objective

Prove that the system respects authorization, resists untrusted content, and supports retention and audit requirements.

### Tasks

- Implement production identity adapter contract and local test identity adapter.
- Implement server-controlled tenant/principal scopes.
- Complete pre-retrieval ACL enforcement across all stores.
- Add data classification, retention, deletion, and model-provider eligibility policies.
- Implement prompt-injection detection/risk signalling.
- Add evidence delimiting and tool allowlists.
- Implement telemetry content sampling/redaction controls.
- Add audit queries and operator documentation.
- Add dependency, container, secret, and static security scanning.
- Build adversarial and authorization datasets.
- Create ADR-007.

### Required tests

- horizontal and vertical privilege attempts;
- inference through graph relationships does not leak forbidden identities;
- malicious instructions in every supported document type;
- deletion across raw storage, vector index, graph, and caches;
- log/trace redaction;
- model-provider eligibility denial;
- rate/size/graph-expansion abuse cases;
- safe error responses.

### Acceptance criteria

- unauthorized retrieval count is zero across the full security dataset;
- deletion/deactivation tests pass across every storage layer;
- no known secret or raw token is present in logs/traces;
- adversarial documents cannot change workflow/tool policy;
- security runbook and threat model are reviewed.

---

## M10 — Operator and review interfaces

### Objective

Provide the minimum interfaces required to operate, inspect, and improve the system safely.

### Tasks

- Build a minimal web UI or server-rendered operator interface.
- Provide chat/query view with source citations.
- Provide evidence viewer with document version and locator.
- Provide safe graph-path viewer.
- Link execution runs to Phoenix traces.
- Provide ingestion status and quality report view.
- Provide entity-resolution/fact verification review queue.
- Provide evaluation run comparison and regression details.
- Provide user feedback capture with category and optional correction.
- Ensure UI never receives evidence the user cannot access.

### Required tests

- API/UI contract tests;
- role/tenant visibility tests;
- citation and locator navigation;
- review decision audit trail;
- accessibility smoke checks;
- UI does not expose raw prompts, secrets, or unauthorized graph details.

### Acceptance criteria

- pilot users can ask, inspect evidence, and submit feedback;
- reviewers can resolve ambiguous entities/facts with an audit trail;
- operators can move from a bad answer to its trace and evaluation evidence;
- all access rules match backend enforcement.

---

## M11 — Production hardening and pilot readiness

### Objective

Prepare a controlled pilot with measured capacity, reliability, cost, rollback, and operating procedures.

### Tasks

- Add environment-specific configuration and secret-manager integration.
- Create production container images and SBOMs.
- Add deployment manifests for the selected target environment.
- Add database backup/restore and migration procedures.
- Add Neo4j backup/rebuild and reconciliation procedures.
- Define retry, circuit-breaker, timeout, and rate-limit policies.
- Load test ingestion and query workloads.
- Measure latency budgets by stage and optimize the actual bottlenecks.
- Add cost budgets and model routing only if evaluation proves quality is retained.
- Run failure injection for provider, PostgreSQL, Neo4j, object storage, and telemetry outages.
- Implement canary/shadow evaluation and rollback.
- Complete operations, incident, data recovery, and reindex runbooks.
- Run full pre-release evaluation and human sample review.

### Required tests

- load and soak tests against representative data volume;
- backup and restore drill;
- failed migration rollback or forward-fix drill;
- dependency failure and recovery;
- telemetry outage does not corrupt business flow;
- partial ingestion recovery;
- canary rollback;
- complete evaluation and security suites.

### Acceptance criteria

- SLOs are met at expected pilot load;
- recovery and rollback are demonstrated, not only documented;
- cost per answer is measured and within the agreed budget;
- all release gates pass;
- pilot has named owners, support procedure, and feedback/review cadence.

---

## M12 — Post-MVP controlled improvement loop

### Objective

Use execution and evaluation data to propose improvements without unsafe autonomous self-modification.

### Tasks

- Cluster failures by ingestion, entity, retrieval, graph, context, generation, citation, policy, and tool categories.
- Generate improvement proposals linked to failing traces and evaluation cases.
- Allow proposals for prompt, threshold, routing, chunking, ontology, and code changes.
- Require offline evaluation for every proposal.
- Require human approval before canary deployment.
- Apply graduated rollout: local -> offline full eval -> shadow -> canary -> production.
- Preserve baseline, proposal, evaluation, approval, deployment, and rollback records.
- Add drift detection for query mix, retrieval scores, graph quality, judge agreement, latency, and cost.

### Explicit prohibition

The production system must not automatically modify prompts, ontology, code, tools, model routing, or thresholds based only on live feedback.

### Acceptance criteria

- every improvement is evidence-linked and reversible;
- no proposal can bypass offline evaluation or approval;
- canary and rollback records are queryable;
- drift alerts lead to review tasks, not automatic production mutation.

---

## 14. Test strategy

### 14.1 Test pyramid

**Unit tests**

- pure domain policies;
- chunking and normalization;
- score fusion;
- ontology validation;
- metric calculations;
- state transitions;
- citation mapping;
- routing rules.

**Contract tests**

- LLM structured outputs;
- provider adapters;
- repository ports;
- API request/response schemas;
- telemetry attribute contracts.

**Integration tests**

- PostgreSQL/pgvector;
- Neo4j;
- object storage;
- outbox worker;
- OpenTelemetry collector/Phoenix export;
- parser libraries.

**End-to-end tests**

- source ingestion to cited answer;
- vector/graph/hybrid comparison;
- historical query;
- source update and supersession;
- refusal and conflict handling;
- authorization isolation;
- replay.

**Evaluation tests**

- golden datasets;
- adversarial datasets;
- baseline comparison;
- judge calibration;
- latency and cost.

### 14.2 External model tests

- Unit tests use deterministic fakes that mimic provider contracts, not simplistic mocks that bypass validation.
- A smaller recorded-contract suite verifies request/response parsing without sending secrets to fixtures.
- Live provider tests are opt-in locally and scheduled in a controlled CI environment.
- Evaluation results must identify whether live or replayed model responses were used.

### 14.3 Coverage

Set an initial repository line-coverage gate of 85%, but do not optimize for coverage alone. Require direct tests for security, authorization, temporal rules, provenance, retries, failure branches, and migrations regardless of coverage percentage.

---

## 15. Configuration and versioning

### 15.1 Configuration groups

- environment and service URLs;
- database/object-store credentials;
- model provider and aliases;
- embedding configuration;
- chunking configuration;
- retrieval budgets;
- entity-resolution bands;
- telemetry export and content policy;
- evaluation thresholds;
- authorization mode;
- retention policies.

### 15.2 Version bundle

Every answer and evaluation must be reproducible from a bundle containing:

```text
code_commit
workflow_version
prompt_bundle_version
ontology_version
document_index_version
embedding_model_version
reranker_version
generation_model_version
evaluation_dataset_version
judge_version
```

Model aliases such as `extractor-small`, `planner`, `generator`, and `judge` belong in versioned configuration. Do not scatter provider model IDs through application code.

---

## 16. Definition of Done

A feature is done only when:

- implementation respects dependency boundaries;
- input/output contracts are typed and validated;
- happy path and relevant failure/security paths are tested;
- database/schema changes have migrations;
- required spans, metrics, and safe diagnostic attributes exist;
- no prohibited content appears in telemetry;
- evaluation impact is measured when retrieval, graph, prompt, model, or answer behaviour changes;
- documentation and ADRs are updated;
- lint, type check, tests, and applicable eval gates pass;
- rollback or disablement is possible for risky changes.

A milestone is done only when all its acceptance criteria pass and its box in the progress ledger is checked.

---

## 17. First implementation sequence

The coding agent should begin with this exact sequence:

1. Create the repository structure and `pyproject.toml`.
2. Add `uv`, Ruff, Pyright, pytest, coverage, and CI.
3. Add typed settings and a minimal FastAPI application.
4. Add Docker Compose for PostgreSQL/pgvector, Neo4j, MinIO, Phoenix, OpenTelemetry Collector, Prometheus, and Grafana.
5. Emit and verify one correlated request trace and one metric.
6. Add initial ADRs and README commands.
7. Run `make check` and record results.
8. Update M0 and M1 progress only when their independent acceptance criteria pass.
9. Continue with domain contracts and migrations in M2.

Do not begin by implementing a chat UI or a large prompt. The first vertical slice is:

```text
one document
-> parsed and versioned chunks
-> vector retrieval
-> evidence-only answer
-> valid citation
-> complete trace
-> one deterministic evaluation case
```

The second vertical slice adds:

```text
two related entities
-> one provenance-backed fact
-> constrained graph path
-> hybrid answer
-> vector-versus-hybrid evaluation comparison
```

These slices validate the architecture before scale and feature breadth are added.

---

## 18. Final handoff requirements

At the end of each implementation session, the coding agent must report:

1. milestone and tasks completed;
2. files and migrations changed;
3. architecture or ADR decisions made;
4. commands executed and their results;
5. evaluation change versus baseline;
6. known limitations or failing cases;
7. the next smallest executable task.

The report must distinguish verified results from assumptions. It must not mark a milestone complete solely because code was written.
