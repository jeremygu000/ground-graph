"""Unit tests for outbox emitter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.infrastructure.postgres.outbox_emitter import OutboxEmitter


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, event: object) -> None:
        self.added.append(event)


@pytest.mark.asyncio
async def test_emit_mention_adds_event() -> None:
    """emit_mention adds an entity_mentioned event to the session."""
    session = _FakeSession()
    emitter = OutboxEmitter(cast(AsyncSession, session))

    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="PostgreSQL",
        candidate_type="Database",
        extraction_confidence=0.9,
    )

    await emitter.emit_mention(mention)

    assert len(session.added) == 1
    event = session.added[0]
    assert event.event_type == "entity_mentioned"
    assert event.aggregate_type == "mention"


@pytest.mark.asyncio
async def test_emit_entity_adds_event() -> None:
    """emit_entity adds an entity_resolved event to the session."""
    session = _FakeSession()
    emitter = OutboxEmitter(cast(AsyncSession, session))

    entity = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Database",
        canonical_name="PostgreSQL",
        aliases=["PostgreSQL"],
    )

    await emitter.emit_entity(entity)

    assert len(session.added) == 1
    event = session.added[0]
    assert event.event_type == "entity_resolved"
    assert event.aggregate_type == "entity"


@pytest.mark.asyncio
async def test_emit_fact_adds_event() -> None:
    """emit_fact adds a fact_candidate event to the session."""
    session = _FakeSession()
    emitter = OutboxEmitter(cast(AsyncSession, session))

    fact = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=uuid4(),
        predicate="depends_on",
        object_id=uuid4(),
        status="candidate",
        confidence=0.8,
        evidence_ids=[],
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["eng"],
    )

    await emitter.emit_fact(fact)

    assert len(session.added) == 1
    event = session.added[0]
    assert event.event_type == "fact_candidate"
    assert event.aggregate_type == "fact"
