"""Unit tests for the evidence-only answer generator."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from groundgraph.domain.retrieval import Evidence, RetrievalPlan
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
    async def test_generate_supported_claim_with_citations(self) -> None:
        """Happy path: LLM returns a supported claim that maps to a real
        evidence_id; the resulting response has a citation pointing to
        the exact chunk@version locator."""
        ev_id = uuid4()
        chunk_id = uuid4()
        version_id = uuid4()
        json_resp = (
            '{"answer": "GraphRAG combines vectors and graphs", "claims": ['
            f'{{"text": "GraphRAG combines vectors and graphs",'
            f' "evidence_ids": ["{ev_id}"], "factual": true,'
            f' "support_status": "supported"}}'
            "]}"
        )
        client = _FakeOpenAIClient(json_resp)
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=ev_id,
                source_id=uuid4(),
                document_id=uuid4(),
                chunk_id=chunk_id,
                content="the evidence content",
                retrieval_method="vector",
                allowed_principals=["eng"],
            )
        ]
        # Evidence has no version_id; the citation locator falls back to
        # the evidence_id, which must still be a string.
        object.__setattr__(evidence[0], "version_id", version_id)

        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["what is GraphRAG?"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("what is GraphRAG?", evidence, plan)

        assert result.status == "answered"
        assert result.answer is not None
        assert len(result.claims) == 1
        assert len(result.citations) == 1
        assert result.citations[0].locator == f"{chunk_id}@{version_id}"
        assert result.citations[0].evidence_id == ev_id

    @pytest.mark.asyncio
    async def test_fabricated_evidence_id_fails_closed(self) -> None:
        """LLM claims a supported claim but references an evidence_id that
        is not in our evidence set; answer must be downgraded to
        insufficient_evidence (no leaked assertion)."""
        json_resp = (
            '{"answer": "Made-up claim", "claims": ['
            '{"text": "Made-up claim",'
            ' "evidence_ids": ["00000000-0000-0000-0000-000000000000"],'
            ' "factual": true, "support_status": "supported"}'
            "]}"
        )
        client = _FakeOpenAIClient(json_resp)
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="real evidence",
                retrieval_method="vector",
                allowed_principals=["eng"],
            )
        ]
        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["q"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("q", evidence, plan)

        assert result.status == "insufficient_evidence"
        assert result.answer is None
        assert result.citations == []
        assert result.warnings, "warning should explain the refusal"

    @pytest.mark.asyncio
    async def test_no_evidence_short_circuits(self) -> None:
        """Empty evidence list refuses without ever calling the LLM."""
        client = _FakeOpenAIClient("unused")
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["q"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("q", [], plan)

        assert result.status == "insufficient_evidence"
        assert result.answer is None
        assert result.confidence_band == "low"

    @pytest.mark.asyncio
    async def test_insufficient_evidence_returns_low_confidence(self) -> None:
        """Even with evidence passed in, if the LLM produces no answer the
        status must be insufficient_evidence."""
        json_resp = '{"answer": "", "claims": []}'
        client = _FakeOpenAIClient(json_resp)
        provider = EvidenceOnlyAnswerGenerator(client=client)  # type: ignore[arg-type]

        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="some evidence",
                retrieval_method="vector",
                allowed_principals=["eng"],
            )
        ]
        plan = RetrievalPlan(
            strategy="vector",
            question_type="fact",
            query_texts=["q"],
            vector_top_k=10,
            final_evidence_limit=5,
        )

        result = await provider.generate("q", evidence, plan)

        assert result.status == "insufficient_evidence"
        assert result.confidence_band == "low"
