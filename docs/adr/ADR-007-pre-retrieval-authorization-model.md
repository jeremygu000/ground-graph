# ADR-007 Pre-retrieval Authorization Model

> **Status**: Accepted
> **Date**: 2026-09-07
> **Deciders**: Plan author
> **Related**: ADR-001, ADR-004, plan.md §M9

## Context

The system must enforce access control **before** any retrieval occurs, on every data store, without exception. Three attack surfaces exist:

1. **Tenant isolation** — user A in tenant A must never see tenant B's data
2. **Principal-level ACL** — within a tenant, principals may only access data they are authorized for
3. **Prompt injection** — malicious instructions embedded in user queries must not escape the retrieval boundary

## Decision

### 1. Authorization happens before retrieval

Every retrieval method (vector, graph, keyword) must accept a `filters` parameter that encodes the caller's `tenant_id` and `principal`. These filters are applied **inside** the store implementation, not at the application layer. This ensures there is no way to bypass ACL by calling the store directly.

```python
# Every retriever must enforce:
filters = {
    "tenant_id": identity.tenant_id,
    "principal": identity.principal,  # or derived ACL list
}
```

### 2. Pre-retrieval identity extraction

`get_identity()` (api/dependencies.py) extracts the trusted `Identity` from request headers. This is the **only** place where tenant/principal is derived from untrusted input. All downstream code receives a validated `Identity` object.

The `Identity` model carries:
- `tenant_id` — required, non-empty string
- `principal` — required, non-empty string

### 3. Prompt injection detection

`PromptInjectionDetector.check()` (application/security/prompt_injection.py) runs on every user question before it enters the workflow:

| Risk level | Action |
|---|---|
| `high` | Block the request; return 400 |
| `medium` | Log and flag in response warnings |
| `low` | Log only |

High-risk patterns include: instruction override attempts (`ignore all previous instructions`), developer mode activations, opaque encoded payloads (`${jndi:}`), and HTML injection (`<script>`, `<svg>`).

### 4. ACL metadata on facts

`KnowledgeFact` carries `allowed_principals: list[str]`. A query from `principal="user-B"` can only access facts where `"user-B" in allowed_principals or "*" in allowed_principals`.

The `allowed_principals` field is populated at ingestion time based on the document's data classification. The **repository** (not the retriever) is responsible for filtering.

### 5. Cross-tenant access is 404

Any `ExecutionRun`, `CanonicalEntity`, or `KnowledgeFact` accessed with a mismatched `tenant_id` returns HTTP 404 (not 403), to avoid tenant enumeration attacks.

## Consequences

**Easier:**
- Authorization is enforced at the storage layer, making bypass impossible
- Prompt injection detection is a simple heuristic that catches common patterns
- 404-for-mismatched-tenant prevents information leakage

**Harder:**
- Every new retriever must implement ACL filtering correctly
- Heuristic prompt injection detection has false positives and false negatives
- Performance cost of ACL filtering on every retrieval call

**Accepted trade-offs:**
- The heuristic detector is a fallback; a dedicated LLM-based classifier is recommended for production (per M9 tasks)
- The 404 behavior may make debugging harder; this is intentional for security

## References

- `src/groundgraph/api/dependencies.py:get_identity` — identity extraction
- `src/groundgraph/application/security/prompt_injection.py` — injection detection
- `src/groundgraph/domain/knowledge.py:KnowledgeFact` — `tenant_id` and `allowed_principals` fields
- `src/groundgraph/api/execution.py` — cross-tenant execution returns 404
- plan.md §M9 (governance, security, adversarial testing)
