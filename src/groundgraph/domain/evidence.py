"""Outbox-event contracts for the Postgres→Neo4j projection.

The transactional outbox is the boundary between PostgreSQL
(write-side source of truth) and Neo4j (graph projection).
The application writes an outbox row in the SAME transaction
as the data it refers to; a worker consumes outbox rows and
projects them to Neo4j idempotently. See plan.md §5.3.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from groundgraph.domain.defaults import empty_json_dict
from groundgraph.domain.types import JsonValue, validate_json_value


class OutboxEventType(StrEnum):
    """Kinds of outbox events the projection worker understands.

    Adding a new event type MUST come with a worker branch in
    M7+ and a migration in M2 phase 1.
    """

    DOCUMENT_PARSED = "document_parsed"
    CHUNKS_CREATED = "chunks_created"
    ENTITY_MENTIONED = "entity_mentioned"
    ENTITY_RESOLVED = "entity_resolved"
    FACT_CANDIDATE = "fact_candidate"


class OutboxEventStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class OutboxEvent(BaseModel):
    """One row of the transactional outbox.

    The worker queries by ``status="pending"`` ordered by
    ``created_at`` and atomically claims a batch via
    ``UPDATE ... SET status='claimed' WHERE id IN (...) AND status='pending'``
    (see-and-claim pattern). ``payload`` is opaque to the database
    but typed by the producer in application code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    aggregate_type: str  # "document" | "chunk" | "entity" | "fact"
    aggregate_id: UUID
    event_type: OutboxEventType
    payload: dict[str, JsonValue] = Field(default_factory=empty_json_dict)
    status: OutboxEventStatus = OutboxEventStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None
    created_at: datetime
    claimed_at: datetime | None = None
    completed_at: datetime | None = None

    @field_validator("payload", mode="before")
    @classmethod
    def _validate_payload(cls, value: object) -> object:
        return validate_json_value(value)
