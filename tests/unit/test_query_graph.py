"""Unit tests for query workflow graph."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.application.answering.claim_validator import DeterministicClaimValidator
from groundgraph.application.ports import (
    AnswerGenerator,
    EvidenceReranker,
    GraphRepository,
    IndexVersionResolver,
    KeywordRetrieverPort,
    RetrievalPlanner,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.hybrid_retrieval import HybridRetrievalResult
from groundgraph.application.settings import Settings
from groundgraph.domain.retrieval import AnswerClaim, Evidence, QueryResponse, RetrievalPlan
from groundgraph.workflows.query_graph import (
    QueryWorkflow,
    QueryWorkflowConfig,
    QueryWorkflowState,
)


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

    async def query_with_evidence(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        index_name: str | None = None,
    ) -> HybridRetrievalResult:
        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="Test evidence",
                retrieval_method="vector",
                allowed_principals=["user1"],
            )
        ]
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="Test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        return HybridRetrievalResult(response=response, evidence=evidence)


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


@pytest.mark.asyncio
async def test_workflow_validate_node_downgrades_unsupported_claims() -> None:
    """Validation downgrades claims whose evidence is missing and sets insufficient_evidence."""
    validator = DeterministicClaimValidator()
    missing_id = uuid4()
    claim = AnswerClaim(
        claim_id=uuid4(),
        text="Some claim.",
        factual=True,
        evidence_ids=[missing_id],
        support_status="supported",
    )
    response = QueryResponse(
        execution_run_id=uuid4(),
        answer="Some claim.",
        status="answered",
        claims=[claim],
        citations=[],
        confidence_band="medium",
        warnings=[],
    )
    evidence: list[Evidence] = []

    validated = await validator.validate(response, evidence)

    unsupported = [c for c in validated.claims if c.factual and c.support_status == "unsupported"]
    assert len(unsupported) == 1


def test_decide_route_insufficient_evidence_triggers_retry() -> None:
    """State with insufficient_evidence status routes to retry when retries remain."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=QueryResponse(
            execution_run_id=uuid4(),
            answer=None,
            status="insufficient_evidence",
            claims=[],
            citations=[],
            confidence_band="low",
            warnings=[],
        ),
    )
    assert state.retries < state.max_retries


def test_decide_route_routes_clarification_to_done() -> None:
    """State with clarification_required status routes to done."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=QueryResponse(
            execution_run_id=uuid4(),
            answer=None,
            status="clarification_required",
            claims=[],
            citations=[],
            confidence_band="medium",
            warnings=[],
        ),
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "done"


def test_decide_route_routes_response_failed_to_fail() -> None:
    """State with response.status='failed' routes to fail."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=QueryResponse(
            execution_run_id=uuid4(),
            answer=None,
            status="failed",
            claims=[],
            citations=[],
            confidence_band="low",
            warnings=["error"],
        ),
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "fail"


@pytest.mark.asyncio
async def test_validate_node_fails_closed_when_unsupported_factual_claims_remain() -> None:
    """Unsupported factual claims trigger insufficient_evidence (fail-closed)."""
    missing_id = uuid4()
    claim = AnswerClaim(
        claim_id=uuid4(),
        text="Unsupported factual claim.",
        factual=True,
        evidence_ids=[missing_id],
        support_status="supported",
    )
    response = QueryResponse(
        execution_run_id=uuid4(),
        answer="Some answer.",
        status="answered",
        claims=[claim],
        citations=[],
        confidence_band="medium",
        warnings=[],
    )
    evidence: list[Evidence] = []

    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=response,
        evidence=evidence,
    )
    result = await workflow._validate_node(state)
    validated: QueryResponse = result["response"]
    assert validated.status == "insufficient_evidence"
    assert validated.answer is None
    assert len(validated.claims) == 0


def test_workflow_state_update_captures_evidence() -> None:
    """State can store evidence list returned from query_with_evidence."""
    evidence = [
        Evidence(
            evidence_id=uuid4(),
            source_id=uuid4(),
            content="Test content",
            retrieval_method="vector",
            allowed_principals=["user1"],
        )
    ]
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        evidence=evidence,
    )
    assert len(state.evidence) == 1


def test_decide_route_routes_answered_to_done() -> None:
    """State with answered status routes to done."""
    run_id = uuid4()
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=QueryResponse(
            execution_run_id=run_id,
            answer="Test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        ),
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "done"


def test_decide_route_routes_error_with_no_retries_left_to_fail() -> None:
    """Error state with no retries remaining routes to fail."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        error="connection refused",
        retries=1,
        max_retries=1,
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "fail"


def test_decide_route_routes_insufficient_evidence_to_retry() -> None:
    """State with insufficient_evidence routes to retry when retries < max_retries."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        response=QueryResponse(
            execution_run_id=uuid4(),
            answer=None,
            status="insufficient_evidence",
            claims=[],
            citations=[],
            confidence_band="low",
            warnings=[],
        ),
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "retry"


def test_decide_route_routes_error_with_retries_left_to_retry() -> None:
    """Error state with retries remaining routes to retry."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        error="timeout",
        retries=0,
        max_retries=1,
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    route = workflow._decide_route(state)
    assert route == "retry"


def test_retry_node_increments_retries() -> None:
    """Retry node increments the retry counter."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        retries=0,
        max_retries=1,
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    result = workflow._retry_node(state)
    assert result["retries"] == 1


def test_fail_node_produces_failed_response() -> None:
    """Fail node produces a QueryResponse with failed status."""
    state = QueryWorkflowState(
        question="test",
        principal="user1",
        tenant_id="tenant1",
        error="permanent failure",
    )
    config = QueryWorkflowConfig(
        session_factory=object(),
        planner=cast(RetrievalPlanner, _FakePlanner()),
        embedding_provider=object(),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        settings=None,
    )
    workflow = QueryWorkflow(config)
    result = workflow._fail_node(state)
    failed_response: QueryResponse = result["response"]
    assert failed_response.status == "failed"
    assert failed_response.answer is None
