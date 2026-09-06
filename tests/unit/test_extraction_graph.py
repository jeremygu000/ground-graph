"""Unit tests for extraction graph."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.workflows.extraction_graph import (
    ExtractionState,
    build_extraction_graph,
)


def test_extraction_state_done_is_false_by_default() -> None:
    """ExtractionState.done() is False when no error."""
    state = ExtractionState(chunk_id=uuid4(), chunk_content="test", tenant_id="t1")
    assert state.done() is False


def test_extraction_state_done_is_true_with_error() -> None:
    """ExtractionState.done() is True when error is set."""
    state = ExtractionState(chunk_id=uuid4(), chunk_content="test", tenant_id="t1", error="boom")
    assert state.done() is True


class _FakeEmitter:
    def __init__(self) -> None:
        self.emitted_mentions: list[object] = []
        self.emitted_entities: list[object] = []
        self.emitted_facts: list[object] = []

    async def emit_mention(self, mention: object) -> None:
        self.emitted_mentions.append(mention)

    async def emit_entity(self, entity: object) -> None:
        self.emitted_entities.append(entity)

    async def emit_fact(self, fact: object) -> None:
        self.emitted_facts.append(fact)


@pytest.mark.asyncio
async def test_build_and_invoke_extraction_graph() -> None:
    """Can build and invoke the extraction graph end-to-end."""

    class _FakeExtractor:
        async def extract(self, text: str, chunk_id: UUID):
            return [
                EntityMention(
                    mention_id=uuid4(),
                    chunk_id=chunk_id,
                    surface_form="PostgreSQL",
                    candidate_type="Database",
                    extraction_confidence=0.9,
                )
            ]

    class _FakeResolver:
        async def resolve(self, mention):
            return CanonicalEntity(
                entity_id=uuid4(),
                entity_type=mention.candidate_type,
                canonical_name=mention.surface_form,
                aliases=[mention.surface_form],
            )

    class _FakeFactExtractor:
        async def extract_facts(self, text, entities, subject_id, evidence_ids):
            return []

    emitter = _FakeEmitter()
    graph = build_extraction_graph(
        entity_extractor=_FakeExtractor(),
        entity_resolver=_FakeResolver(),
        fact_extractor=_FakeFactExtractor(),
        outbox_emitter=emitter,
    )

    state = ExtractionState(
        chunk_id=uuid4(),
        chunk_content="PostgreSQL is a database.",
        tenant_id="test-tenant",
    )

    result = await graph.ainvoke(state)
    assert len(result["entities"]) == 1
    assert len(result["resolved_entities"]) == 1
    assert len(emitter.emitted_entities) == 1


@pytest.mark.asyncio
async def test_extraction_graph_ambiguous_entity() -> None:
    """Entities that resolve to None go to ambiguous list."""

    class _FakeExtractor:
        async def extract(self, text: str, chunk_id: UUID):
            return [
                EntityMention(
                    mention_id=uuid4(),
                    chunk_id=chunk_id,
                    surface_form="AmbiguousName",
                    candidate_type="Concept",
                    extraction_confidence=0.5,
                )
            ]

    class _FakeResolver:
        async def resolve(self, mention):
            return None

    class _FakeFactExtractor:
        async def extract_facts(self, text, entities, subject_id, evidence_ids):
            return []

    emitter = _FakeEmitter()
    graph = build_extraction_graph(
        entity_extractor=_FakeExtractor(),
        entity_resolver=_FakeResolver(),
        fact_extractor=_FakeFactExtractor(),
        outbox_emitter=emitter,
    )

    state = ExtractionState(
        chunk_id=uuid4(),
        chunk_content="AmbiguousName is ambiguous.",
        tenant_id="test-tenant",
    )

    result = await graph.ainvoke(state)
    assert len(result["entities"]) == 1
    assert len(result["ambiguous_mentions"]) == 1
    assert len(result["resolved_entities"]) == 0


@pytest.mark.asyncio
async def test_extraction_graph_emits_facts() -> None:
    """Facts extracted from content are emitted via outbox."""

    class _FakeExtractor:
        async def extract(self, text: str, chunk_id: UUID):
            return [
                EntityMention(
                    mention_id=uuid4(),
                    chunk_id=chunk_id,
                    surface_form="AuthService",
                    candidate_type="Service",
                    extraction_confidence=0.9,
                ),
                EntityMention(
                    mention_id=uuid4(),
                    chunk_id=chunk_id,
                    surface_form="Redis",
                    candidate_type="Database",
                    extraction_confidence=0.9,
                ),
            ]

    class _FakeResolver:
        async def resolve(self, mention):
            return CanonicalEntity(
                entity_id=uuid4(),
                entity_type=mention.candidate_type,
                canonical_name=mention.surface_form,
                aliases=[mention.surface_form],
            )

    class _FakeFactExtractor:
        async def extract_facts(self, text, entities, name_to_id, evidence_ids):
            redis_id = name_to_id.get("Redis", uuid4())
            auth_id = name_to_id.get("AuthService", uuid4())
            return [
                KnowledgeFact(
                    fact_id=uuid4(),
                    subject_id=auth_id,
                    predicate="depends_on",
                    object_id=redis_id,
                    status="candidate",
                    confidence=0.8,
                    evidence_ids=evidence_ids,
                    observed_at=datetime.now(UTC),
                    extraction_method="llm",
                    ontology_version="v0.1.0",
                )
            ]

    emitter = _FakeEmitter()
    graph = build_extraction_graph(
        entity_extractor=_FakeExtractor(),
        entity_resolver=_FakeResolver(),
        fact_extractor=_FakeFactExtractor(),
        outbox_emitter=emitter,
    )

    state = ExtractionState(
        chunk_id=uuid4(),
        chunk_content="AuthService depends on Redis.",
        tenant_id="test-tenant",
    )

    result = await graph.ainvoke(state)
    assert len(result["facts"]) == 1
    assert len(emitter.emitted_facts) == 1
