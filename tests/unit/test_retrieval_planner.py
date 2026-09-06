"""Unit tests for retrieval planner."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.application.retrieval.retrieval_planner import (
    RetrievalPlanner,
    RetrievalPlannerConfig,
)
from groundgraph.domain.retrieval import ResolvedEntity


class _FakeExtractor:
    async def extract(self, text: str, chunk_id):
        return []


class _FakeResolver:
    async def resolve(self, mention):
        return None


class _FakeEmbed:
    model = "test-model"
    dimensions = 1536

    async def embed_one(self, text: str):
        return [0.1] * 1536

    async def embed(self, texts: list[str]):
        return [[0.1] * 1536] * len(texts)


def _make_planner() -> RetrievalPlanner:
    config = RetrievalPlannerConfig(
        entity_extractor=_FakeExtractor(),
        entity_resolver=_FakeResolver(),
        embedding_provider=_FakeEmbed(),
    )
    return RetrievalPlanner(config=config)


@pytest.fixture
def planner() -> RetrievalPlanner:
    return _make_planner()


@pytest.mark.asyncio
async def test_plan_temporal_question(planner: RetrievalPlanner) -> None:
    """Questions with 'when'/'history' keywords are classified as temporal."""
    plan = await planner.plan("When was AuthService first deployed?", "user1", "tenant1")
    assert plan.question_type == "temporal"
    assert plan.strategy in ("vector", "graph", "hybrid")


@pytest.mark.asyncio
async def test_plan_relationship_question(planner: RetrievalPlanner) -> None:
    """Questions about dependencies are classified as relationship."""
    plan = await planner.plan("What services does AuthService depend on?", "user1", "tenant1")
    assert plan.question_type == "relationship"
    assert plan.strategy in ("graph", "hybrid")


@pytest.mark.asyncio
async def test_plan_multi_hop(planner: RetrievalPlanner) -> None:
    """Plans have valid strategy and graph depth configuration."""
    plan = await planner.plan(
        "What connects the database to the authentication layer?", "user1", "tenant1"
    )
    assert plan.question_type in ("fact", "relationship", "multi_hop", "summary")
    assert plan.max_graph_depth >= 1


@pytest.mark.asyncio
async def test_plan_fact_question(planner: RetrievalPlanner) -> None:
    """Simple 'how'/'why' questions are classified as fact."""
    plan = await planner.plan("How does authentication work in this system?", "user1", "tenant1")
    assert plan.question_type == "fact"
    assert plan.strategy == "vector"


@pytest.mark.asyncio
async def test_plan_comparison_question(planner: RetrievalPlanner) -> None:
    """Questions with 'vs'/'versus' are classified as comparison."""
    plan = await planner.plan("PostgreSQL vs MongoDB for this use case", "user1", "tenant1")
    assert plan.question_type == "comparison"
    assert plan.strategy in ("hybrid", "graph")


@pytest.mark.asyncio
async def test_plan_reason_codes_present(planner: RetrievalPlanner) -> None:
    """Plans include explainable reason codes."""
    plan = await planner.plan("What services does AuthService depend on?", "user1", "tenant1")
    assert len(plan.reason_codes) > 0
    assert any("q_type=" in code for code in plan.reason_codes)
    assert any("strategy=" in code for code in plan.reason_codes)


def test_select_strategy_relationship_with_entities() -> None:
    """Relationship questions with entities use hybrid strategy."""
    p = _make_planner()
    e = ResolvedEntity(
        entity_id=uuid4(),
        canonical_name="test",
        entity_type="Service",
    )
    strategy = p._select_strategy("relationship", [e])
    assert strategy == "hybrid"


def test_select_strategy_relationship_without_entities() -> None:
    """Relationship questions without entities use graph strategy."""
    p = _make_planner()
    strategy = p._select_strategy("relationship", [])
    assert strategy == "graph"


def test_select_strategy_comparison() -> None:
    """Comparison questions use hybrid strategy."""
    p = _make_planner()
    strategy = p._select_strategy("comparison", [])
    assert strategy == "hybrid"


def test_select_strategy_fact() -> None:
    """Fact questions use vector strategy."""
    p = _make_planner()
    strategy = p._select_strategy("fact", [])
    assert strategy == "vector"


def test_graph_depth_for_type() -> None:
    """Graph depth varies by question type."""
    p = _make_planner()
    assert p._graph_depth_for_type("fact") == 1
    assert p._graph_depth_for_type("relationship") == 2
    assert p._graph_depth_for_type("multi_hop") == 3
    assert p._graph_depth_for_type("temporal") == 1
    assert p._graph_depth_for_type("impact") == 2
    assert p._graph_depth_for_type("unknown") == 1
