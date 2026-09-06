"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.domain.retrieval import Evidence
from groundgraph.infrastructure.openai.reranker import CrossEncoderReranker


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _FakeCompletions:
    def __init__(self, response_content: str) -> None:
        self._response_content = response_content

    async def create(self, **kwargs: object) -> _FakeCompletionResponse:
        return _FakeCompletionResponse(self._response_content)


class _FakeChat:
    def __init__(self, response_content: str) -> None:
        self._response_content = response_content

    @property
    def completions(self) -> _FakeCompletions:
        return _FakeCompletions(self._response_content)


class _FakeOpenAIClient:
    def __init__(self, response_content: str) -> None:
        self._response_content = response_content
        self.chat = _FakeChat(self._response_content)

    async def close(self) -> None:
        pass


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_reranker_sorts_by_score(self) -> None:
        client = _FakeOpenAIClient("4")
        reranker = CrossEncoderReranker(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="irrelevant content",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="very relevant content about the question",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
        ]

        result = await reranker.rerank(evidence, "what is this about")

        assert len(result) == 2
        assert result[0].rerank_score == 4.0
        assert result[1].rerank_score == 4.0

    @pytest.mark.asyncio
    async def test_single_evidence_returns_unchanged(self) -> None:
        """Single evidence is returned unchanged (no reranking needed)."""
        client = _FakeOpenAIClient("5")
        reranker = CrossEncoderReranker(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="some content",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
        ]

        result = await reranker.rerank(evidence, "question")

        assert len(result) == 1
        assert result[0].content == "some content"

    @pytest.mark.asyncio
    async def test_invalid_score_defaults_to_one(self) -> None:
        """Invalid LLM score response defaults to 1.0."""
        client = _FakeOpenAIClient("not a number")
        reranker = CrossEncoderReranker(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="content",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="more content",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
        ]

        result = await reranker.rerank(evidence, "question")

        assert len(result) == 2
        for r in result:
            assert r.rerank_score == 1.0
