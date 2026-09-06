"""Unit tests for the evidence-only answer generator."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan
from groundgraph.infrastructure.openai.answer_generator import EvidenceOnlyAnswerGenerator


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    async def create(self, **kwargs: Any) -> _FakeCompletionResponse:
        return _FakeCompletionResponse(self._content)


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self._content = content

    @property
    def completions(self) -> _FakeCompletions:
        return _FakeCompletions(self._content)


class _FakeOpenAIClient:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChat(content)

    async def close(self) -> None:
        pass


class TestEvidenceOnlyAnswerGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_query_response(self) -> None:
        client = _FakeOpenAIClient('{"answer": "Test answer", "claims": [], "status": "answered"}')
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="some evidence",
                retrieval_method="vector",
                allowed_principals=["eng"],
            ),
        ]

        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["what is this?"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("what is this?", evidence, plan)

        assert isinstance(result, QueryResponse)
        assert result.answer == "Test answer"
        assert result.status == "answered"

    @pytest.mark.asyncio
    async def test_insufficient_evidence_returns_low_confidence(self) -> None:
        json_resp = '{"answer": "", "claims": [], "status": "insufficient_evidence"}'
        client = _FakeOpenAIClient(json_resp)
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        evidence: list[Evidence] = []
        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["what is this?"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("what is this?", evidence, plan)

        assert result.status == "insufficient_evidence"
        assert result.confidence_band == "low"
