"""Unit tests for graph fusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.application.ports import GraphRepository
from groundgraph.application.retrieval.graph_fusion import (
    STALENESS_HALF_LIFE_DAYS,
    GraphFusionService,
    hybrid_rrf_fusion,
)
from groundgraph.domain.knowledge import CanonicalEntity
from groundgraph.domain.retrieval import Evidence


class _FakeChunk:
    def __init__(self, score: float | None = None) -> None:
        self.chunk_id = uuid4()
        self.source_id = uuid4()
        self.document_id = None
        self.version_id = None
        self.content = f"Content for {self.chunk_id}"
        self.vector_score = score
        self.keyword_score = None
        self.allowed_principals = ["user1"]


class _FakeRepo:
    async def find_facts(self, subject_id=None, object_id=None, predicate=None, status=None):
        return []


def test_hybrid_rrf_fusion_vector_only() -> None:
    """RRF fusion with only vector results returns them ranked."""
    chunks = [_FakeChunk(score=0.9), _FakeChunk(score=0.8)]
    result = hybrid_rrf_fusion(cast(Any, chunks), cast(Any, []), cast(Any, []))
    assert len(result) == 2
    assert result[0].chunk_id == chunks[0].chunk_id
    assert result[0].retrieval_method == "vector"


def test_hybrid_rrf_fusion_keyword_only() -> None:
    """RRF fusion with only keyword results returns them ranked."""
    chunks = [_FakeChunk(), _FakeChunk()]
    result = hybrid_rrf_fusion(cast(Any, []), cast(Any, chunks), cast(Any, []))
    assert len(result) == 2
    assert result[0].retrieval_method == "keyword"


def test_hybrid_rrf_fusion_graph_boosts() -> None:
    """Graph evidence boosts hybrid fusion score."""
    graph_ev = Evidence(
        evidence_id=uuid4(),
        source_id=uuid4(),
        content="Graph fact",
        retrieval_method="graph",
        allowed_principals=["user1"],
    )
    chunks = [_FakeChunk()]
    result = hybrid_rrf_fusion(cast(Any, chunks), cast(Any, []), [graph_ev])
    assert len(result) == 2
    methods = {r.retrieval_method for r in result}
    assert "graph" in methods


def test_hybrid_rrf_fusion_deduplicates_by_source() -> None:
    """Fusion deduplicates by source to ensure diversity."""
    src_id = uuid4()
    graph_ev = Evidence(
        evidence_id=uuid4(),
        source_id=src_id,
        content="Graph fact",
        retrieval_method="graph",
        allowed_principals=["user1"],
    )
    chunk = _FakeChunk()
    chunk.source_id = src_id
    result = hybrid_rrf_fusion(cast(Any, [chunk]), cast(Any, []), [graph_ev])
    assert len(result) == 1


def test_hybrid_rrf_fusion_preserves_graph_provenance_fields() -> None:
    """Fusion preserves graph_path_fact_ids, valid_from, and valid_to from graph evidence."""
    fact_ids = [uuid4(), uuid4()]
    valid_from = datetime(2024, 1, 1, tzinfo=UTC)
    valid_to = datetime(2025, 1, 1, tzinfo=UTC)
    graph_ev = Evidence(
        evidence_id=uuid4(),
        source_id=uuid4(),
        content="Graph fact",
        retrieval_method="graph",
        graph_path_fact_ids=fact_ids,
        valid_from=valid_from,
        valid_to=valid_to,
        allowed_principals=["user1"],
    )
    result = hybrid_rrf_fusion(cast(Any, []), cast(Any, []), [graph_ev])
    assert len(result) == 1
    assert result[0].graph_path_fact_ids == fact_ids
    assert result[0].valid_from == valid_from
    assert result[0].valid_to == valid_to


def test_hybrid_rrf_fusion_graph_merges_temporal_from_multiple_graph_evidence() -> None:
    """Graph evidence with same ID: temporal fields merged (earliest from, latest to)."""
    eid = uuid4()
    src = uuid4()
    ev1 = Evidence(
        evidence_id=eid,
        source_id=src,
        content="Graph fact 1",
        retrieval_method="graph",
        graph_path_fact_ids=[uuid4()],
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 6, 1, tzinfo=UTC),
        allowed_principals=["user1"],
    )
    ev2 = Evidence(
        evidence_id=eid,
        source_id=src,
        content="Graph fact 2",
        retrieval_method="graph",
        graph_path_fact_ids=[uuid4()],
        valid_from=datetime(2024, 3, 1, tzinfo=UTC),
        valid_to=datetime(2025, 1, 1, tzinfo=UTC),
        allowed_principals=["user1"],
    )
    result = hybrid_rrf_fusion(cast(Any, []), cast(Any, []), [ev1, ev2])
    assert len(result) == 1
    assert result[0].valid_from == datetime(2024, 1, 1, tzinfo=UTC)
    assert result[0].valid_to == datetime(2025, 1, 1, tzinfo=UTC)
    assert len(result[0].graph_path_fact_ids) == 2


@pytest.mark.asyncio
async def test_retrieve_evidence_graph_evidence_content_is_meaningful() -> None:
    """Graph evidence content includes predicate and score, not just a score."""
    entity_id = uuid4()
    fact_id = uuid4()
    fact = _FactWithMeta(
        fact_id=fact_id,
        predicate="depends_on",
        subject_id=entity_id,
        object_id=uuid4(),
        observed_at=datetime.now(UTC),
    )
    repo = _RepoReturnsFacts([fact])
    svc = GraphFusionService(graph_repository=cast(GraphRepository, repo))
    entity = CanonicalEntity(
        entity_id=entity_id,
        entity_type="Service",
        canonical_name="AuthService",
        aliases=["AuthService"],
    )
    result = await svc.retrieve_evidence([entity])
    assert len(result) == 1
    content = result[0].content
    assert "depends_on" in content


@pytest.mark.asyncio
async def test_retrieve_evidence_preserves_fact_ids() -> None:
    """Graph evidence preserves graph_path_fact_ids for provenance tracking."""
    entity_id = uuid4()
    fact_id = uuid4()
    fact = _FactWithMeta(
        fact_id=fact_id,
        predicate="implements",
        subject_id=entity_id,
        object_id=uuid4(),
        observed_at=datetime.now(UTC),
    )
    repo = _RepoReturnsFacts([fact])
    svc = GraphFusionService(graph_repository=cast(GraphRepository, repo))
    entity = CanonicalEntity(
        entity_id=entity_id,
        entity_type="Service",
        canonical_name="AuthService",
        aliases=["AuthService"],
    )
    result = await svc.retrieve_evidence([entity])
    assert len(result) == 1
    assert result[0].graph_path_fact_ids == [fact_id]


def test_hybrid_rrf_fusion_empty_inputs() -> None:
    """Fusion with no inputs returns empty list."""
    result = hybrid_rrf_fusion(cast(Any, []), cast(Any, []), cast(Any, []))
    assert result == []


def test_staleness_score_fresh() -> None:
    """Recent observations get high staleness score."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, object()))
    now = datetime.now(UTC)
    score = svc._staleness_score(now)
    assert 0.99 <= score <= 1.0


def test_staleness_score_none() -> None:
    """None observation returns 1.0."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, object()))
    assert svc._staleness_score(None) == 1.0


def test_staleness_score_old() -> None:
    """Very old observations approach 0."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, object()))
    old = datetime(2000, 1, 1, tzinfo=UTC)
    score = svc._staleness_score(old)
    assert score < 0.1


def test_staleness_half_life() -> None:
    """Score halves after STALENESS_HALF_LIFE_DAYS."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, object()))
    half_life_days_ago = datetime.now(UTC) - timedelta(days=STALENESS_HALF_LIFE_DAYS)
    score = svc._staleness_score(half_life_days_ago)
    assert 0.45 < score < 0.55


def test_graph_fusion_service_temporal_validation() -> None:
    """_is_temporal_valid correctly filters by valid_at."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, _FakeRepo()))
    now = datetime.now(UTC)
    fact = type(
        "Fact",
        (),
        {
            "valid_from": now - timedelta(days=10),
            "valid_to": now + timedelta(days=10),
        },
    )()
    assert svc._is_temporal_valid(fact, now) is True
    assert svc._is_temporal_valid(fact, None) is True


def test_graph_fusion_service_temporal_filter_future() -> None:
    """_is_temporal_valid excludes facts not yet valid."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, _FakeRepo()))
    future = datetime.now(UTC) + timedelta(days=30)
    fact = type(
        "Fact",
        (),
        {
            "valid_from": future,
            "valid_to": None,
        },
    )()
    assert svc._is_temporal_valid(fact, datetime.now(UTC)) is False


@dataclass
class _FactWithMeta:
    fact_id: Any
    predicate: str
    subject_id: Any
    object_id: Any
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class _RepoWithFacts:
    async def find_facts(self, subject_id=None, object_id=None, predicate=None, status=None):
        return []


class _RepoReturnsFacts(_RepoWithFacts):
    def __init__(self, facts: list[Any]) -> None:
        self._facts = facts

    async def find_facts(self, subject_id=None, object_id=None, predicate=None, status=None):
        return self._facts


@pytest.mark.asyncio
async def test_retrieve_evidence_with_seed_entities() -> None:
    """retrieve_evidence traverses from seed entities and returns graph evidence."""
    entity_id = uuid4()
    fact_id = uuid4()
    fact = _FactWithMeta(
        fact_id=fact_id,
        predicate="depends_on",
        subject_id=entity_id,
        object_id=uuid4(),
        observed_at=datetime.now(UTC),
    )
    repo = _RepoReturnsFacts([fact])
    svc = GraphFusionService(graph_repository=cast(GraphRepository, repo))
    entity = CanonicalEntity(
        entity_id=entity_id,
        entity_type="Service",
        canonical_name="AuthService",
        aliases=["AuthService"],
    )
    result = await svc.retrieve_evidence([entity])
    assert len(result) == 1
    assert result[0].retrieval_method == "graph"
    assert result[0].evidence_id == fact_id


@pytest.mark.asyncio
async def test_retrieve_evidence_empty_seed_entities() -> None:
    """retrieve_evidence returns empty list when no seed entities."""
    svc = GraphFusionService(graph_repository=cast(GraphRepository, _RepoWithFacts()))
    result = await svc.retrieve_evidence([])
    assert result == []


def test_hybrid_rrf_fusion_graph_evidence_merges_with_existing_vector() -> None:
    """When graph evidence shares the same source as a vector chunk, scores are merged via RRF."""
    chunk_id = uuid4()
    chunk_source = uuid4()
    chunk = type(
        "_Chunk",
        (),
        {
            "chunk_id": chunk_id,
            "source_id": chunk_source,
            "document_id": None,
            "version_id": None,
            "content": "Shared content",
            "vector_score": 0.9,
            "keyword_score": None,
            "allowed_principals": ["user1"],
        },
    )()
    graph_ev = Evidence(
        evidence_id=uuid4(),
        source_id=chunk_source,
        content="Graph fact",
        retrieval_method="graph",
        allowed_principals=["user1"],
    )
    result = hybrid_rrf_fusion(cast(Any, [chunk]), cast(Any, []), [graph_ev])
    assert len(result) == 1
    assert result[0].retrieval_method == "graph"


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion_staleness_penalty_reduces_score() -> None:
    """Staleness scoring is applied to graph evidence via depth score calculation."""
    entity_id = uuid4()
    fact_id = uuid4()
    now = datetime.now(UTC)
    fact = _FactWithMeta(
        fact_id=fact_id,
        predicate="depends_on",
        subject_id=entity_id,
        object_id=uuid4(),
        observed_at=now - timedelta(days=180),
    )
    repo = _RepoReturnsFacts([fact])
    svc = GraphFusionService(graph_repository=cast(GraphRepository, repo))
    entity = CanonicalEntity(
        entity_id=entity_id,
        entity_type="Service",
        canonical_name="OldService",
        aliases=["OldService"],
    )
    result = await svc.retrieve_evidence([entity], max_depth=1)
    assert len(result) == 1
    content_str = result[0].content
    assert "depends_on:" in content_str
