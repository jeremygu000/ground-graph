"""Unit tests for keyword retriever spans and result construction."""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from groundgraph.infrastructure.postgres.keyword_retriever import PostgresKeywordRetriever


class _FakeSession:
    async def execute(self, stmt: Any) -> Any:
        return _FakeResult()


class _FakeResult:
    def all(self) -> list[Any]:
        return [_FakeRow()]


class _FakeRow:
    def __init__(  # noqa: PLR0917
        self,
        chunk_id: Any = None,
        source_id: Any = None,
        document_id: Any = None,
        version_id: Any = None,
        content: str = "test content",
        rank: float = 0.5,
        allowed_principals: list[str] | None = None,
    ) -> None:
        self.chunk_id = chunk_id or uuid4()
        self.source_id = source_id or uuid4()
        self.document_id = document_id or uuid4()
        self.version_id = version_id or uuid4()
        self.content = content
        self.rank = rank
        self.allowed_principals = allowed_principals or ["eng"]


@pytest.mark.asyncio
async def test_keyword_search_returns_results() -> None:
    session = cast(Any, _FakeSession())
    retriever = PostgresKeywordRetriever(session)
    results = await retriever.search(
        "test query",
        top_k=5,
        tenant_id="test-tenant",
        allowed_principals=["eng"],
    )
    assert len(results) == 1
    assert results[0].content == "test content"
    assert results[0].rank == 0.5


@pytest.mark.asyncio
async def test_keyword_search_empty_query_returns_empty() -> None:
    session = cast(Any, _FakeSession())
    retriever = PostgresKeywordRetriever(session)
    results = await retriever.search(
        "",
        top_k=5,
        tenant_id="test-tenant",
        allowed_principals=["eng"],
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_keyword_search_whitespace_query_returns_empty() -> None:
    session = cast(Any, _FakeSession())
    retriever = PostgresKeywordRetriever(session)
    results = await retriever.search(
        "   ",
        top_k=5,
        tenant_id="test-tenant",
        allowed_principals=["eng"],
    )
    assert len(results) == 0
