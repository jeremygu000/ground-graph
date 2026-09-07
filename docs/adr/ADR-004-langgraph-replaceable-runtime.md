# ADR-004 LangGraph as Replaceable Workflow Runtime

> **Status**: Accepted
> **Date**: 2026-09-06
> **Deciders**: Plan author
> **Related**: ADR-001, ADR-008, plan.md §7.2

## Context

The query workflow (`QueryWorkflow`) requires complex, multi-step orchestration with:

- **State** — must carry question, principal, tenant, plan, response, evidence, warnings, retry count, error
- **Branching** — the workflow decides at each step whether to retry, fail, or complete based on claim validation
- **Checkpointing** — long-running executions must survive API server restarts
- **Audit** — every state transition must be recorded for replay

We evaluated three options for the orchestration layer:

1. **Custom async state machine** — pure Python, explicit transitions
2. **LangGraph** — `StateGraph` with `MemorySaver` checkpointing
3. **LangChain** — higher-level abstractions but heavier coupling

LangGraph is the practical choice for the initial implementation (M7) because it provides checkpointed state persistence and a clean node/edge model out of the box. However, it must remain a **replaceable** dependency to satisfy the architecture rule that domain and application layers must not import framework types.

## Decision

### 1. LangGraph is an infrastructure dependency only

`src/groundgraph/workflows/query_graph.py` imports `langgraph` at the **workflow layer** (not domain or application). All application-layer services (`HybridRetrievalService`, `DeterministicClaimValidator`, etc.) are instantiated outside LangGraph and passed in via the `QueryWorkflowConfig` dataclass.

```
application  ←  workflows  ←  langgraph (framework import boundary)
```

### 2. Workflow interface is framework-agnostic

`QueryWorkflow` exposes only `async ainvoke(question, principal, tenant_id, index_name) -> dict`:

```python
class QueryWorkflow:
    async def ainvoke(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        index_name: str | None = None,
    ) -> QueryResponse: ...
```

No LangGraph types appear in the public interface. The internal `_build_graph()` method constructs the `StateGraph`, but callers never see it.

### 3. Checkpointing uses MemorySaver (not a database)

The checkpointer is `MemorySaver()` — an in-memory, per-process checkpoint store. This means:

- **Replay works** within a single server process (acceptable for MVP)
- **Cross-process replay** requires replacing `MemorySaver` with a persistent checkpointer (future work)
- The `ExecutionRepository` (PostgreSQL) provides durable replay via stored `ExecutionRun` + `ExecutionStep` records

### 4. Runtime is swappable

Because LangGraph is instantiated inside `QueryWorkflow.__init__` and the workflow exposes only an `ainvoke` interface, replacing LangGraph with an alternative runtime (e.g., a custom `asyncio.Task` based state machine, or a Pregel-compatible engine) requires only:

1. A new class implementing the same `ainvoke` signature
2. Replacing `QueryWorkflow` instantiation in `_build_query_workflow(dependencies.py)`

No application-layer or domain-layer code changes are required.

## Consequences

**Easier:**
- Checkpointed state handles retries without re-executing已完成 steps
- LangGraph's `MemorySaver` provides zero-config persistence for development
- Node/edge model makes the workflow logic explicit and auditable

**Harder:**
- LangGraph is a medium-weight dependency (significant transitive dependencies)
- `MemorySaver` is in-memory only; server restart loses checkpoint state
- LangGraph error messages can be verbose; we wrap them in `WorkflowError`

**Accepted trade-offs:**
- LangGraph is a direct dependency (not transitive); replacing it is a one-file change in the workflow layer
- The in-memory checkpoint limitation is acceptable because `ExecutionRepository` provides durable replay at the API layer

## Alternatives considered

**Custom async state machine:**
Rejected because implementing checkpointing, retry bounds, and node routing from scratch is high effort and would duplicate LangGraph's battle-tested logic.

**LangChain:**
Rejected because LangChain has heavier coupling to OpenAI and more restrictive abstractions; LangGraph is lower-level and gives us more control.

## References

- `src/groundgraph/workflows/query_graph.py` — the `QueryWorkflow` implementation
- `src/groundgraph/api/dependencies.py:_build_query_workflow` — LangGraph instantiation site
- plan.md §7.2 (query workflow acceptance criteria)
