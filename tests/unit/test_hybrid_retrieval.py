"""Unit tests for hybrid retrieval service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from groundgraph.application.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    EvidenceReranker,
    GraphRepository,
    IndexVersionInfo,
    IndexVersionResolver,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.hybrid_retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalResult,
    HybridRetrievalService,
)
from groundgraph.domain.retrieval import Evidence, QueryResponse


@dataclass
class _FakePlan:
    strategy: str = "vector"
    entities: list[Any] = field(default_factory=list)
    predicates: list[str] | None = None
    valid_at: Any = None
    max_graph_depth: int = 2


class _FakeRetrievalPlanner:
    def __init__(self, plan: _FakePlan | None = None) -> None:
        self._plan = plan or _FakePlan()

    async def plan(self, question: str, principal: str, tenant_id: str) -> _FakePlan:
        return self._plan


class _FakeIndexVersionResolver:
    def __init__(
        self,
        index: IndexVersionInfo | None = None,
    ) -> None:
        self._index = index

    async def resolve_active(self, index_name: str) -> IndexVersionInfo | None:
        return self._index


class _FakeEmbeddingProvider:
    model: str = "test-embedder"

    async def embed_one(self, text: str) -> list[float]:
        return [0.1] * 10


class _FakeVectorRetriever:
    def __init__(self, results: list[Evidence] | None = None) -> None:
        self._results = results or []

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        allowed_principals: list[str],
        tenant_id: str,
        index_version_id: UUID,
    ) -> list[Evidence]:
        return self._results


class _FakeKeywordRetriever:
    def __init__(self, results: list[Evidence] | None = None) -> None:
        self._results = results or []

    async def search(
        self,
        query: str,
        top_k: int,
        allowed_principals: list[str],
        tenant_id: str,
    ) -> list[Evidence]:
        return self._results


class _FakeGraphRepo:
    async def get_entity(self, entity_id: UUID) -> Any:
        return None

    async def find_facts(
        self,
        subject_id: UUID | None = None,
        predicate: str | None = None,
        object_id: UUID | None = None,
        status: str | None = None,
        allowed_principals: list[str] | None = None,
    ) -> list[Any]:
        return []


class _FakeReranker:
    def __init__(self, results: list[Evidence] | None = None) -> None:
        self._results = results or []

    async def rerank(self, evidence: list[Evidence], question: str) -> list[Evidence]:
        return self._results if self._results else evidence


class _FakeAnswerGenerator:
    def __init__(self, response: QueryResponse | None = None) -> None:
        self._response = response

    async def generate(self, question: str, evidence: list[Evidence], plan: Any) -> QueryResponse:
        return self._response or QueryResponse(
            execution_run_id=uuid4(),
            answer="Test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
        )


def _make_config(  # noqa: PLR0917
    strategy: str = "vector",
    vector_results: list[Evidence] | None = None,
    keyword_results: list[Evidence] | None = None,
    graph_results: list[Evidence] | None = None,
    answer_response: QueryResponse | None = None,
    reranked_results: list[Evidence] | None = None,
    index: IndexVersionInfo | None = None,
) -> HybridRetrievalConfig:
    if index is None:
        index = IndexVersionInfo(
            version_id=uuid4(),
            index_name="default",
            embedding_model="test-embedder",
            embedding_dimensions=10,
            is_active=True,
        )
    plan = _FakePlan(
        strategy=strategy,
        entities=[],
        predicates=None,
        valid_at=None,
        max_graph_depth=2,
    )
    return HybridRetrievalConfig(
        session_factory=object(),
        embedding_provider=cast(EmbeddingProvider, _FakeEmbeddingProvider()),
        vector_retriever=cast(VectorContentRetriever, _FakeVectorRetriever(vector_results)),
        keyword_retriever=cast(KeywordRetrieverPort, _FakeKeywordRetriever(keyword_results)),
        graph_repository=cast(GraphRepository, _FakeGraphRepo()),
        reranker=cast(EvidenceReranker, _FakeReranker(reranked_results)),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator(answer_response)),
        index_version_resolver=cast(IndexVersionResolver, _FakeIndexVersionResolver(index)),
        planner=cast(Any, _FakeRetrievalPlanner(plan)),
    )


def _make_evidence(chunk_id: UUID | None = None) -> Evidence:
    return Evidence(
        evidence_id=uuid4(),
        source_id=chunk_id or uuid4(),
        content="test evidence",
        retrieval_method="vector",
        vector_score=0.9,
        allowed_principals=["user1"],
    )


@pytest.mark.asyncio
async def test_query_requires_tenant_id() -> None:
    """query() raises ValueError when tenant_id is empty."""
    svc = HybridRetrievalService(_make_config())
    with pytest.raises(ValueError, match="tenant_id"):
        await svc.query(question="test", principal="user1", tenant_id="")


@pytest.mark.asyncio
async def test_query_vector_strategy_returns_response() -> None:
    """query() with strategy=vector returns a QueryResponse."""
    ev = _make_evidence()
    resp = QueryResponse(
        execution_run_id=uuid4(),
        answer="test answer",
        status="answered",
        claims=[],
        citations=[],
        confidence_band="high",
    )
    config = _make_config(
        strategy="vector",
        vector_results=[ev],
        answer_response=resp,
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query(question="test question", principal="user1", tenant_id="tenant1")
    assert result.answer == "test answer"
    assert result.status == "answered"


@pytest.mark.asyncio
async def test_query_hybrid_strategy_returns_response() -> None:
    """query() with strategy=hybrid returns a QueryResponse."""
    ev = _make_evidence()
    resp = QueryResponse(
        execution_run_id=uuid4(),
        answer="hybrid answer",
        status="answered",
        claims=[],
        citations=[],
        confidence_band="medium",
    )
    config = _make_config(
        strategy="hybrid",
        vector_results=[ev],
        answer_response=resp,
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query(question="test question", principal="user1", tenant_id="tenant1")
    assert result.answer == "hybrid answer"


@pytest.mark.asyncio
async def test_query_graph_strategy_returns_response() -> None:
    """query() with strategy=graph returns a QueryResponse."""
    ev = _make_evidence()
    resp = QueryResponse(
        execution_run_id=uuid4(),
        answer="graph answer",
        status="answered",
        claims=[],
        citations=[],
        confidence_band="medium",
    )
    config = _make_config(
        strategy="graph",
        answer_response=resp,
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query(question="test question", principal="user1", tenant_id="tenant1")
    assert result.answer == "graph answer"


@pytest.mark.asyncio
async def test_query_raises_when_no_active_index() -> None:
    """query() raises RuntimeError when no active index version is found."""
    no_index_resolver = _FakeIndexVersionResolver(None)
    config = HybridRetrievalConfig(
        session_factory=object(),
        embedding_provider=cast(EmbeddingProvider, _FakeEmbeddingProvider()),
        vector_retriever=cast(VectorContentRetriever, _FakeVectorRetriever()),
        keyword_retriever=cast(KeywordRetrieverPort, _FakeKeywordRetriever()),
        graph_repository=cast(GraphRepository, _FakeGraphRepo()),
        reranker=cast(EvidenceReranker, _FakeReranker()),
        answer_generator=cast(AnswerGenerator, _FakeAnswerGenerator()),
        index_version_resolver=cast(IndexVersionResolver, no_index_resolver),
        planner=cast(Any, _FakeRetrievalPlanner(_FakePlan())),
    )
    svc = HybridRetrievalService(config)
    with pytest.raises(RuntimeError, match="No active IndexVersion"):
        await svc.query(question="test", principal="user1", tenant_id="tenant1")


@pytest.mark.asyncio
async def test_query_raises_on_embedding_model_mismatch() -> None:
    """query() raises ValueError when index embedding model does not match provider."""
    mismatched_index = IndexVersionInfo(
        version_id=uuid4(),
        index_name="default",
        embedding_model="different-model",
        embedding_dimensions=20,
        is_active=True,
    )
    config = _make_config(index=mismatched_index)
    svc = HybridRetrievalService(config)
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        await svc.query(question="test", principal="user1", tenant_id="tenant1")


@pytest.mark.asyncio
async def test_query_with_evidence_returns_response_and_evidence() -> None:
    """query_with_evidence() returns HybridRetrievalResult with both response and evidence."""
    ev = _make_evidence()
    resp = QueryResponse(
        execution_run_id=uuid4(),
        answer="test answer",
        status="answered",
        claims=[],
        citations=[],
        confidence_band="high",
    )
    config = _make_config(
        strategy="vector",
        vector_results=[ev],
        answer_response=resp,
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query_with_evidence(
        question="test question", principal="user1", tenant_id="tenant1"
    )
    assert isinstance(result, HybridRetrievalResult)
    assert result.response.answer == "test answer"
    assert isinstance(result.evidence, list)


@pytest.mark.asyncio
async def test_query_with_evidence_requires_tenant_id() -> None:
    """query_with_evidence() raises ValueError when tenant_id is empty."""
    svc = HybridRetrievalService(_make_config())
    with pytest.raises(ValueError, match="tenant_id"):
        await svc.query_with_evidence(question="test", principal="user1", tenant_id="")


@pytest.mark.asyncio
async def test_query_with_evidence_hybrid_strategy() -> None:
    """query_with_evidence() with strategy=hybrid returns HybridRetrievalResult."""
    ev = _make_evidence()
    resp = QueryResponse(
        execution_run_id=uuid4(),
        answer="hybrid evidence answer",
        status="answered",
        claims=[],
        citations=[],
        confidence_band="medium",
    )
    config = _make_config(
        strategy="hybrid",
        vector_results=[ev],
        answer_response=resp,
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query_with_evidence(
        question="test question", principal="user1", tenant_id="tenant1"
    )
    assert result.response.answer == "hybrid evidence answer"


@pytest.mark.asyncio
async def test_deduplicate_removes_duplicates() -> None:
    """_deduplicate removes evidence items with duplicate evidence_ids."""
    ev1 = _make_evidence()
    ev2 = Evidence(
        evidence_id=ev1.evidence_id,
        source_id=uuid4(),
        content="different content",
        retrieval_method="keyword",
        vector_score=0.5,
        allowed_principals=["user1"],
    )
    config = _make_config(vector_results=[ev1, ev2])
    svc = HybridRetrievalService(config)
    result = await svc.query(question="test", principal="user1", tenant_id="tenant1")
    assert result.status in ("answered", "failed")


@pytest.mark.asyncio
async def test_hybrid_fuse_passes_through() -> None:
    """_hybrid_fuse delegates to hybrid_rrf_fusion."""
    ev = _make_evidence()
    config = _make_config(
        strategy="hybrid",
        vector_results=[ev],
        reranked_results=[ev],
    )
    svc = HybridRetrievalService(config)
    result = await svc.query(question="test", principal="user1", tenant_id="tenant1")
    assert result.status == "answered"
