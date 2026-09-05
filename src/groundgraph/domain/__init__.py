"""Domain layer — pure Pydantic v2 types, no infrastructure imports.

See AGENTS.md §4: domain types must not import SQLAlchemy, Neo4j,
FastAPI, LangGraph, OpenAI, or Phoenix. Only ``pydantic``,
``datetime``, ``uuid``, ``enum``, and stdlib are allowed.
"""

from groundgraph.domain.documents import (
    Chunk,
    ParsedDocument,
    SourceDescriptor,
)
from groundgraph.domain.evidence import (
    OutboxEvent,
    OutboxEventStatus,
    OutboxEventType,
)
from groundgraph.domain.execution import (
    ALLOWED_RUN_TRANSITIONS,
    ExecutionRun,
    ExecutionRunStatus,
    ExecutionStep,
    ExecutionStepStatus,
    assert_run_transition,
)
from groundgraph.domain.knowledge import (
    CanonicalEntity,
    EntityMention,
    KnowledgeFact,
)
from groundgraph.domain.retrieval import (
    AnswerClaim,
    Citation,
    Evidence,
    QueryResponse,
    ResolvedEntity,
    RetrievalPlan,
)
from groundgraph.domain.types import JsonValue

__all__ = [
    "ALLOWED_RUN_TRANSITIONS",
    "AnswerClaim",
    "CanonicalEntity",
    "Chunk",
    "Citation",
    "EntityMention",
    "Evidence",
    "ExecutionRun",
    "ExecutionRunStatus",
    "ExecutionStep",
    "ExecutionStepStatus",
    "JsonValue",
    "KnowledgeFact",
    "OutboxEvent",
    "OutboxEventStatus",
    "OutboxEventType",
    "ParsedDocument",
    "QueryResponse",
    "ResolvedEntity",
    "RetrievalPlan",
    "SourceDescriptor",
    "assert_run_transition",
]
