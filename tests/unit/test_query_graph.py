"""Unit tests for query workflow graph."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from groundgraph.application.ports import (
    AnswerGenerator,
    EvidenceReranker,
    GraphRepository,
    IndexVersionResolver,
    KeywordRetrieverPort,
    RetrievalPlanner,
    VectorContentRetriever,
)
from groundgraph.application.settings import Settings
from groundgraph.domain.retrieval import QueryResponse, RetrievalPlan
from groundgraph.workflows.query_graph import QueryWorkflowConfig, QueryWorkflowState


class _FakeRetrievalPlan:
    strategy: str = "vector"
    question_type: str = "fact"
    query_texts: list[str] | None = None
    entities: list | None = None
    predicates: Any = None
    max_graph_depth: int = 2
    vector_top_k: int = 10
    final_evidence_limit: int = 10
    valid_at: Any = None
    reason_codes: list | None = None


class _FakePlanner:
    async def plan(self, question: str, principal: str, tenant_id: str) -> RetrievalPlan:
        return _FakeRetrievalPlan()  # type: ignore[return-value]


class _FakeAnswerGenerator:
    async def generate(
        self, question: str, evidence: list, retrieval_plan: RetrievalPlan
    ) -> QueryResponse:
        return QueryResponse(
            execution_run_id=uuid4(),
            answer="Test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )


class _FakeHybridSvc:
    async def query(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        index_name: str | None = None,
    ) -> QueryResponse:
        return _FakeAnswerGenerator().generate("", [], _FakeRetrievalPlan())


class _FakeQueryWorkflowConfig:
    session_factory: Any = None
    planner: RetrievalPlanner = cast(RetrievalPlanner, _FakePlanner())
    embedding_provider: Any = None
    vector_retriever: VectorContentRetriever = cast(VectorContentRetriever, object())
    keyword_retriever: KeywordRetrieverPort = cast(KeywordRetrieverPort, object())
    graph_repository: GraphRepository = cast(GraphRepository, object())
    reranker: EvidenceReranker = cast(EvidenceReranker, object())
    answer_generator: AnswerGenerator = cast(AnswerGenerator, _FakeAnswerGenerator())
    index_version_resolver: IndexVersionResolver = cast(IndexVersionResolver, object())
    settings: Settings | None = None


def test_workflow_state_defaults() -> None:
    """QueryWorkflowState has correct defaults."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
    )
    assert state.retries == 0
    assert state.max_retries == 1
    assert state.evidence == []
    assert state.response is None


def test_workflow_config_can_be_instantiated() -> None:
    """QueryWorkflowConfig accepts required arguments."""
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, object()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    assert config.planner is not None
