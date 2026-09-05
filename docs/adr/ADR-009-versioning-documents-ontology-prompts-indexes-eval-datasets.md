# ADR-009 Versioning of Documents, Ontology, Prompts, Indexes, and Evaluation Datasets

> **Status**: Accepted
> **Date**: 2026-09-05
> **Deciders**: Plan author
> **Related**: ADR-003, plan.md §4.4, §5.1, §6.3, §11.2

## Context

A production RAG system must be reproducible: the answer a user received last week
was generated from a specific document version, a specific embedding index, a specific
prompt template, and a specific model. If any of these change, previously correct
answers may become incorrect or a regression may go undetected.

Furthermore, for evaluation and debugging, the system must be able to answer:
> "Show me the inputs and config that produced execution run X."

Without explicit versioning of every configurable artifact, this is impossible.

## Decision

Every artifact that influences an answer or evaluation score is versioned independently:

| Artifact | Version Record | Immutable? | Storage |
|---|---|---|---|
| Document content | `document_versions` row with `checksum` | Yes — append-only | PostgreSQL + S3 |
| Embedding index | `index_versions` row | Yes for historical; latest is active | PostgreSQL |
| Ontology | `ontology_version` string in settings + YAML in repo | Yes — tag in repo | YAML files |
| Prompt template | `prompt_versions` row | Yes — append-only | PostgreSQL |
| Model config | `model_config_versions` row | Yes — append-only | PostgreSQL |
| Evaluation dataset | `evaluation_datasets` row | Yes — dataset is frozen | PostgreSQL |

### Document versioning rules

- Same source URI + same checksum → no new version (idempotent re-ingest)
- Changed content → new `document_versions` row with new `version_id`, old row preserved
- "Deleting" a source sets `is_current=False` on the latest version; data is never deleted
- Reprocessing with different parser/chunker/embedding model → new `index_version` (not new document)

### Index versioning rules

- `index_versions` records: `embedding_model`, `embedding_dimensions`, `chunker_version`
- A new index version is created when any of these change
- Only one `index_version` per `index_name` can be `is_active=True`
- Search MUST filter by `index_version` to avoid mixing embeddings from different models

### Prompt/model config versioning rules

- Every prompt bundle or model config change creates a new version row
- `prompt_bundle_version` in settings references the active bundle version
- Execution runs record which version was used at invocation time

### Evaluation dataset versioning rules

- Datasets are append-only; a new dataset version is created for meaningful changes
- Each `evaluation_case` belongs to exactly one `evaluation_dataset`
- Evaluation results reference `dataset_id`, `run_id`, and all artifact versions used

## Consequences

**Easier:**
- Full reproducibility of any execution run
- Regression detection: compare evaluation scores across prompt/index/ontology versions
- Audit trail for compliance
- Safe reprocessing: same content always maps to same document version

**Harder:**
- Schema requires more tables and foreign keys
- Outbox events must carry version IDs
- Query context must include `valid_at` for temporal queries and `index_version` for retrieval

**Accepted trade-offs:**
- Storage cost for immutable versions is acceptable given S3 + PostgreSQL economics
- Schema complexity is managed by SQLAlchemy models and Alembic migrations

## Alternatives considered

**Git-based versioning only:**
Store all artifacts in git. Rejected because binary documents, large evaluation datasets, and runtime model configs do not belong in git.

**Single global version number:**
Increment one "system version" on any change. Rejected because it conflates unrelated changes and makes isolated regression analysis impossible.

**No document immutability:**
Overwrite on re-ingest. Rejected because it destroys the audit trail and makes it impossible to answer "what was true at time X".

## References

- plan.md §4.4 (structured-output policy)
- plan.md §5.1 (PostgreSQL tables)
- plan.md §6.3 (idempotency and versioning)
- plan.md §11.2 (evaluation dataset schema)
