"""Unit tests for the Neo4j graph repository adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import (
    Neo4jGraphRepository,
    _dict_to_entity,
    _dict_to_fact,
    _neo4j_datetime_to_native,
)


@dataclass
class _FakeResult:
    data_rows: list[dict[str, Any]] | None = None
    single_row: dict[str, Any] | None = None

    async def data(self) -> list[dict[str, Any]]:
        return self.data_rows or []

    async def single(self) -> dict[str, Any] | None:
        return self.single_row


class _FakeTx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeTx:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def run(self, query: str, **params: Any) -> _FakeResult:
        self._session.runs.append((query, params))
        return self._session.response

    async def commit(self) -> None:
        self._session.commits += 1


class _FakeSession:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, Any]]] = []
        self.commits = 0
        self.response: _FakeResult = _FakeResult()

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def begin_transaction(self) -> _FakeTx:
        return _FakeTx(self)

    async def run(self, query: str, **params: Any) -> _FakeResult:
        self.runs.append((query, params))
        return self.response

    async def execute_read(self, func: Any) -> Any:
        return await func(self)

    async def execute_write(self, func: Any) -> Any:
        return await func(self)


class _FakeSessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeDriver:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.session_calls = 0

    def session(self) -> _FakeSessionContext:
        self.session_calls += 1
        return _FakeSessionContext(self._session)


def _repo(session: _FakeSession) -> Neo4jGraphRepository:
    return Neo4jGraphRepository(cast(Any, _FakeDriver(session)))


def test_helper_functions_round_trip() -> None:
    now = datetime.now(UTC)
    entity = _dict_to_entity(
        {
            "e": {
                "entity_id": uuid4(),
                "entity_type": "Service",
                "canonical_name": "API",
                "aliases": ["api"],
                "attributes": '{"region":"us-east-1"}',
            }
        }
    )
    assert isinstance(entity, CanonicalEntity)
    assert _neo4j_datetime_to_native(now) == now

    fact = _dict_to_fact(
        {
            "fact_id": uuid4(),
            "subject_id": uuid4(),
            "predicate": "DEPENDS_ON",
            "object_id": uuid4(),
            "status": "verified",
            "confidence": 0.9,
            "evidence_ids": [uuid4()],
            "valid_from": now,
            "valid_to": now + timedelta(microseconds=1),
            "observed_at": now,
            "extraction_method": "structured",
            "ontology_version": "v1",
        }
    )
    assert isinstance(fact, KnowledgeFact)


@pytest.mark.asyncio
async def test_create_get_find_entity_and_mention() -> None:
    session = _FakeSession()
    repo = _repo(session)
    entity = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="API Gateway",
        aliases=["api"],
        attributes={"region": "us-east-1"},
    )

    await repo.create_entity(entity)
    session.response = _FakeResult(
        single_row={
            "e": {
                "entity_id": str(entity.entity_id),
                "entity_type": entity.entity_type,
                "canonical_name": entity.canonical_name,
                "aliases": entity.aliases,
                "attributes": '{"region":"us-east-1"}',
            }
        }
    )
    assert await repo.get_entity(entity.entity_id) is not None
    session.response = _FakeResult(
        data_rows=[
            {
                "e": {
                    "entity_id": str(entity.entity_id),
                    "entity_type": entity.entity_type,
                    "canonical_name": entity.canonical_name,
                    "aliases": entity.aliases,
                    "attributes": '{"region":"us-east-1"}',
                }
            }
        ]
    )
    found = await repo.find_entities(canonical_name="API Gateway")
    assert len(found) == 1

    mention = EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form="api",
        candidate_type="Service",
        extraction_confidence=0.9,
    )
    await repo.create_mention(mention)
    session.response = _FakeResult(
        data_rows=[
            {
                "m": {
                    "mention_id": str(mention.mention_id),
                    "chunk_id": str(mention.chunk_id),
                    "surface_form": mention.surface_form,
                    "candidate_type": mention.candidate_type,
                    "locator": None,
                    "extraction_confidence": mention.extraction_confidence,
                }
            }
        ]
    )
    mentions = await repo.find_mentions(mention.chunk_id)
    assert len(mentions) == 1


@pytest.mark.asyncio
async def test_create_entity_rejects_mutated_attributes() -> None:
    session = _FakeSession()
    repo = _repo(session)
    attributes: dict[str, object] = {"ok": {"nested": "value"}}
    entity = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="API Gateway",
        aliases=["api"],
        attributes=attributes,
    )
    cast(dict[str, object], entity.attributes["ok"])["blob"] = (1, 2)

    with pytest.raises(ValueError, match="JsonValue"):
        await repo.create_entity(entity)

    assert session.runs == []


@pytest.mark.asyncio
async def test_bound_transaction_reuses_same_tx_for_update_and_read() -> None:
    session = _FakeSession()
    driver = _FakeDriver(session)
    tx = await session.begin_transaction()
    repo = Neo4jGraphRepository(cast(Any, driver), tx=cast(Any, tx))
    fact_id = uuid4()
    updated_fact = KnowledgeFact(
        fact_id=fact_id,
        subject_id=uuid4(),
        predicate="DEPENDS_ON",
        object_id=uuid4(),
        status="verified",
        confidence=0.8,
        evidence_ids=[uuid4()],
        valid_from=datetime.now(UTC),
        valid_to=None,
        observed_at=datetime.now(UTC),
        extraction_method="structured",
        ontology_version="v1",
    )
    session.response = _FakeResult(
        single_row={
            "f": {
                "fact_id": updated_fact.fact_id,
                "subject_id": updated_fact.subject_id,
                "predicate": updated_fact.predicate,
                "object_id": updated_fact.object_id,
                "status": updated_fact.status,
                "confidence": updated_fact.confidence,
                "evidence_ids": updated_fact.evidence_ids,
                "valid_from": updated_fact.valid_from,
                "valid_to": updated_fact.valid_to,
                "observed_at": updated_fact.observed_at,
                "extraction_method": updated_fact.extraction_method,
                "ontology_version": updated_fact.ontology_version,
            }
        }
    )

    result = await repo.update_fact_status(fact_id, "verified")

    assert result.status == "verified"
    assert driver.session_calls == 0
    assert session.commits == 0
    assert len(session.runs) == 2
    assert session.runs[0][0].strip().startswith("MATCH (f:Fact")
    assert session.runs[1][0].strip().startswith("MATCH (f:Fact")


@pytest.mark.asyncio
async def test_create_get_find_fact_and_status_update() -> None:
    session = _FakeSession()
    repo = _repo(session)
    subject = uuid4()
    obj = uuid4()
    valid_from = datetime.now(UTC)
    fact = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=subject,
        predicate="DEPENDS_ON",
        object_id=obj,
        status="candidate",
        confidence=0.8,
        evidence_ids=[uuid4()],
        valid_from=valid_from,
        valid_to=valid_from + timedelta(microseconds=1),
        observed_at=valid_from,
        extraction_method="structured",
        ontology_version="v1",
    )

    await repo.create_fact(fact)
    session.response = _FakeResult(
        single_row={
            "f": {
                "fact_id": fact.fact_id,
                "subject_id": subject,
                "predicate": fact.predicate,
                "object_id": obj,
                "status": fact.status,
                "confidence": fact.confidence,
                "evidence_ids": fact.evidence_ids,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "observed_at": fact.observed_at,
                "extraction_method": fact.extraction_method,
                "ontology_version": fact.ontology_version,
            }
        }
    )
    assert await repo.get_fact(fact.fact_id) is not None

    session.response = _FakeResult(
        data_rows=[
            {
                "f": {
                    "fact_id": fact.fact_id,
                    "subject_id": subject,
                    "predicate": fact.predicate,
                    "object_id": obj,
                    "status": fact.status,
                    "confidence": fact.confidence,
                    "evidence_ids": fact.evidence_ids,
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "observed_at": fact.observed_at,
                    "extraction_method": fact.extraction_method,
                    "ontology_version": fact.ontology_version,
                }
            }
        ]
    )
    assert len(await repo.find_facts(subject_id=subject, predicate="DEPENDS_ON")) == 1

    session.response = _FakeResult(
        single_row={
            "f": {
                "fact_id": fact.fact_id,
                "subject_id": subject,
                "predicate": fact.predicate,
                "object_id": obj,
                "status": "verified",
                "confidence": fact.confidence,
                "evidence_ids": fact.evidence_ids,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "observed_at": fact.observed_at,
                "extraction_method": fact.extraction_method,
                "ontology_version": fact.ontology_version,
            }
        }
    )
    updated = await repo.update_fact_status(fact.fact_id, "verified")
    assert updated.status == "verified"
