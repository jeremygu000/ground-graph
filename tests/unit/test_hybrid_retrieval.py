"""Unit tests for hybrid retrieval service."""

from __future__ import annotations

from typing import Any, cast

import pytest

from groundgraph.application.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    EvidenceReranker,
    GraphRepository,
    IndexVersionResolver,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.hybrid_retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalService,
)


class _FakeRetrievalPlanner:
    async def plan(self, question: str, principal: str, tenant_id: str) -> Any:
        return type(
            "Plan",
            (),
            {
                "strategy": "vector",
                "entities": [],
                "predicates": None,
                "valid_at": None,
                "max_graph_depth": 2,
            },
        )()


@pytest.mark.asyncio
async def test_query_requires_tenant_id() -> None:
    """query() raises ValueError when tenant_id is empty."""
    config = HybridRetrievalConfig(
        session_factory=object(),
        embedding_provider=cast(EmbeddingProvider, object()),
        vector_retriever=cast(VectorContentRetriever, object()),
        keyword_retriever=cast(KeywordRetrieverPort, object()),
        graph_repository=cast(GraphRepository, object()),
        reranker=cast(EvidenceReranker, object()),
        answer_generator=cast(AnswerGenerator, object()),
        index_version_resolver=cast(IndexVersionResolver, object()),
        planner=cast(Any, _FakeRetrievalPlanner()),
    )
    svc = HybridRetrievalService(config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tenant_id"):
        await svc.query(question="test", principal="user1", tenant_id="")
