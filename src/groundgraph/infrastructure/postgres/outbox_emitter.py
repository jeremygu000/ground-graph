"""Outbox emitter for knowledge graph events."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.domain.evidence import OutboxEventType
from groundgraph.infrastructure.postgres.models import Outbox


class OutboxEmitter:
    """Emit entity and fact events to the PostgreSQL outbox."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def emit_mention(self, mention: Any) -> None:
        """Emit an ENTITY_MENTIONED event."""
        event = Outbox(
            event_id=uuid4(),
            aggregate_type="mention",
            aggregate_id=mention.mention_id,
            event_type=OutboxEventType.ENTITY_MENTIONED.value,
            payload={
                "mention_id": str(mention.mention_id),
                "chunk_id": str(mention.chunk_id),
                "surface_form": mention.surface_form,
                "candidate_type": mention.candidate_type,
                "locator": mention.locator,
                "extraction_confidence": mention.extraction_confidence,
            },
            status="pending",
        )
        self._session.add(event)

    async def emit_entity(self, entity: Any) -> None:
        """Emit an ENTITY_RESOLVED event."""
        event = Outbox(
            event_id=uuid4(),
            aggregate_type="entity",
            aggregate_id=entity.entity_id,
            event_type=OutboxEventType.ENTITY_RESOLVED.value,
            payload={
                "entity_id": str(entity.entity_id),
                "entity_type": entity.entity_type,
                "canonical_name": entity.canonical_name,
                "aliases": entity.aliases,
                "attributes": entity.attributes,
            },
            status="pending",
        )
        self._session.add(event)

    async def emit_fact(self, fact: Any) -> None:
        """Emit a FACT_CANDIDATE event."""
        event = Outbox(
            event_id=uuid4(),
            aggregate_type="fact",
            aggregate_id=fact.fact_id,
            event_type=OutboxEventType.FACT_CANDIDATE.value,
            payload={
                "fact_id": str(fact.fact_id),
                "subject_id": str(fact.subject_id),
                "predicate": fact.predicate,
                "object_id": str(fact.object_id),
                "status": fact.status,
                "confidence": fact.confidence,
                "evidence_ids": [str(e) for e in fact.evidence_ids],
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
                "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                "observed_at": fact.observed_at.isoformat() if fact.observed_at else None,
                "extraction_method": fact.extraction_method,
                "ontology_version": fact.ontology_version,
            },
            status="pending",
        )
        self._session.add(event)
