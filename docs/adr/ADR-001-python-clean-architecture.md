# ADR-001: Python and clean architecture boundaries

- **Status**: Accepted
- **Date**: 2026-09-04
- **Deciders**: Architecture team
- **Related**: `docs/plan.md` §2.2, §2.3, §4, `AGENTS.md` §4

## Context

The system must integrate many framework libraries (FastAPI, LangGraph, OpenAI,
Neo4j, SQLAlchemy, Phoenix) over a long lifetime. Without strict dependency
boundaries, application logic becomes coupled to one provider, tests become
hard to isolate, and reasoning about authorization and telemetry becomes
unsafe. The plan mandates a layered architecture with explicit
domain/application/infrastructure separation.

## Decision

We adopt the following dependency direction as a binding rule:

```
domain  <-  application  <-  workflows / API  <-  infrastructure composition
```

Concretely:

1. The `domain` package contains only Pydantic v2 data contracts and pure
   policies. It does not import LangGraph, FastAPI, OpenAI, Neo4j,
   SQLAlchemy, Pydantic Settings, Phoenix, or any framework.
2. The `application` package defines use cases and ports (Python
   `Protocol`/`ABC`). It may import from `domain` but not from any
   framework.
3. The `infrastructure` package provides adapters that implement
   application ports and may freely import framework libraries.
4. The `workflows` package contains the LangGraph adapter and may call
   application services. Application services MUST NOT import LangGraph
   types.
5. The `api` package contains FastAPI routers and may call application
   services.
6. Tests may import any layer but use fake/in-memory adapters to keep
   tests fast and isolated.

Pyright `strict` mode is enabled for `domain` and `application` to make
boundary violations type errors at the editor level. Ruff's import-order
rules reinforce the visual layering.

## Consequences

- Application logic is portable across providers and runtime frameworks.
- Tests can substitute fakes for slow or external dependencies.
- Authorization and telemetry code is centralized in infrastructure
  adapters and cannot be bypassed by application code that imports
  directly into databases.
- Pyright `strict` for core layers may surface many existing typing
  gaps; we accept this as the price of a clean boundary.
- New contributors must learn the layering before contributing
  application code. We mitigate this through `AGENTS.md` §4 and a
  lint guard that forbids framework imports inside `domain/`.

## Alternatives considered

- **Layered architecture without enforced type rules**: easier to start,
  decays quickly. Rejected because boundary erosion has been the most
  common long-term cost in similar systems.
- **Hexagonal with explicit port/adapter interface files in every
  module**: more formal but produces excessive boilerplate for a young
  codebase. We use Python `Protocol` typing as a lighter equivalent.
- **Single package with no boundaries**: rejected; explicitly disallowed
  by `plan.md` §0.1 rule 5.

## References

- `docs/plan.md` §2.2, §2.3, §4, §13 (DoD)
- `AGENTS.md` §4
- `pyproject.toml` `[tool.pyright].strict`
