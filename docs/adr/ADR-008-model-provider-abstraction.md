# ADR-008 Model Provider Abstraction and Routing

> **Status**: Accepted
> **Date**: 2026-09-06
> **Deciders**: Plan author
> **Related**: ADR-004, ADR-007, plan.md §8.2

## Context

The system makes LLM calls in two distinct roles:

1. **Embedding** — `EmbeddingProvider.embed()` and `EmbeddingProvider.embed_one()` for vector search
2. **Generation** — `AnswerGenerator.generate()` for producing answers from evidence

Both currently use OpenAI as the sole provider. The architecture requires:

- **Swapability** — changing from OpenAI to Anthropic, Azure OpenAI, or a local model must not require modifying application-layer code
- **Routing** — different request types (retrieval vs. generation) may use different models
- **Version tracking** — model choices must be captured in execution records for replay (ADR-004)

## Decision

### 1. Provider interfaces are defined as Protocol ports

The `EmbeddingProvider` and `AnswerGenerator` are **application-layer ports** (`application/ports.py`). Infrastructure adapters implement these protocols:

| Port | OpenAI Adapter | Alternative |
|---|---|---|
| `EmbeddingProvider` | `OpenAIEmbeddingProvider` | `AzureEmbeddingProvider`, `LocalEmbeddingProvider` |
| `AnswerGenerator` | `EvidenceOnlyAnswerGenerator` | `AzureAnswerGenerator`, `AnthropicAnswerGenerator` |

The `EvidenceOnlyAnswerGenerator` uses `openai` SDK directly and is configured via `Settings`. No OpenAI-specific types appear in application-layer signatures.

### 2. Embedding model is versioned and validated at query time

Per ADR-009, every query validates that `EmbeddingProvider.model` matches the active `IndexVersion.embedding_model` before any retrieval occurs. This prevents cross-version contamination.

The `IndexVersionResolver` is a port (see `application/ports.py:IndexVersionResolver`) that returns `IndexVersionInfo` containing the expected model string and dimensions.

### 3. Generation model is captured at execution time

`Settings.generation_model` is the authoritative model string for answer generation. This value is stored in `ExecutionRun.model_version` at query time, enabling:

- **Replay with model** — replay can optionally use the recorded model
- **A/B routing** — a future model router can select models per-request type
- **Cost attribution** — per-execution model selection enables cost tracking

### 4. Routing is via factory functions

The `get_retrieval_service()` and `_build_query_workflow()` functions in `api/dependencies.py` construct the full adapter stack. To add a new provider:

1. Implement the `Protocol` in `infrastructure/`
2. Change the factory function in `dependencies.py`

No application-layer or domain-layer changes are required.

## Consequences

**Easier:**
- Adding a new provider requires only an infrastructure adapter and a factory change
- Model version tracking enables reproducible replay and cost attribution
- Protocol-based design makes testing easy (fake implementations satisfy the protocol)

**Harder:**
- Each adapter must handle its own retry logic, rate limits, and error translation
- OpenAI-specific features (e.g., structured output via `response_format`) require provider-specific configuration paths

**Accepted trade-offs:**
- Protocol interfaces add some boilerplate but make the swap boundary explicit
- All current providers use OpenAI SDK, so provider-specific behavior is minimal

## References

- `src/groundgraph/application/ports.py:EmbeddingProvider` — embedding protocol
- `src/groundgraph/application/ports.py:AnswerGenerator` — generation protocol
- `src/groundgraph/application/ports.py:IndexVersionResolver` — index version resolution port
- `src/groundgraph/infrastructure/openai/embedding_provider.py` — OpenAI adapter
- `src/groundgraph/infrastructure/openai/answer_generator.py` — OpenAI answer generator
- `src/groundgraph/api/dependencies.py` — adapter factory functions
- `src/groundgraph/domain/execution.py` — `model_version` on `ExecutionRun`
- ADR-009 (versioning of indexes and prompts)
