"""Tests for the DeepEval adapter and JSONL dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from groundgraph.application.evaluation.deepeval_adapter import (
    GroundGraphCitationMetric,
    GroundGraphNoHallucinationMetric,
    load_jsonl_dataset,
)
from groundgraph.domain.retrieval import (
    AnswerClaim,
    Citation,
    Evidence,
    QueryResponse,
)


class TestLoadJsonlDataset:
    """Tests for the JSONL dataset loader."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "test-001",
                    "question": "What is Python?",
                    "tenant_id": "tenant1",
                    "principal": "user1",
                    "expected_status": "answered",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "test-002",
                    "question": "What is Rust?",
                    "tenant_id": "tenant1",
                    "principal": "user2",
                }
            )
            + "\n"
        )
        cases = load_jsonl_dataset(str(path))
        assert len(cases) == 2
        assert cases[0].id == "test-001"
        assert cases[0].question == "What is Python?"
        assert cases[0].expected_status == "answered"
        assert cases[1].id == "test-002"

    def test_load_with_optional_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "test-001",
                    "question": "What is Python?",
                    "tenant_id": "tenant1",
                    "principal": "user1",
                    "required_claim_text": "Python",
                    "forbidden_claim_text": ["Rust", "Go"],
                    "expected_entities": ["Python"],
                }
            )
            + "\n"
        )
        cases = load_jsonl_dataset(str(path))
        assert cases[0].required_claim_text == "Python"
        assert cases[0].forbidden_claim_text == ["Rust", "Go"]
        assert cases[0].expected_entities == ["Python"]

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(json.dumps({"id": "test-001", "question": "What?"}) + "\n")
        with pytest.raises(ValueError, match="missing required field"):
            load_jsonl_dataset(str(path))

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text("")
        cases = load_jsonl_dataset(str(path))
        assert cases == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            "\n\n"
            + json.dumps({"id": "t1", "question": "q1", "tenant_id": "t1", "principal": "p1"})
            + "\n\n"
        )
        cases = load_jsonl_dataset(str(path))
        assert len(cases) == 1


class TestGroundGraphCitationMetric:
    """Tests for citation accuracy metric."""

    @pytest.mark.asyncio
    async def test_all_citations_found(self) -> None:
        eid1, eid2 = uuid4(), uuid4()
        evidence = [
            Evidence(
                evidence_id=eid1,
                source_id=uuid4(),
                content="chunk 1",
                retrieval_method="vector",
            ),
            Evidence(
                evidence_id=eid2,
                source_id=uuid4(),
                content="chunk 2",
                retrieval_method="graph",
            ),
        ]
        citations = [
            Citation(citation_id=uuid4(), claim_id=uuid4(), evidence_id=eid1, locator="[1]"),
            Citation(citation_id=uuid4(), claim_id=uuid4(), evidence_id=eid2, locator="[2]"),
        ]
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="answer",
            status="answered",
            claims=[],
            citations=citations,
            confidence_band="high",
            warnings=[],
        )
        metric = GroundGraphCitationMetric()
        result = await metric.evaluate(response, evidence)
        assert result.success
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_missing_citation_id(self) -> None:
        eid1, eid2 = uuid4(), uuid4()
        evidence = [
            Evidence(
                evidence_id=eid1,
                source_id=uuid4(),
                content="chunk 1",
                retrieval_method="vector",
            ),
        ]
        citations = [
            Citation(citation_id=uuid4(), claim_id=uuid4(), evidence_id=eid2, locator="[1]"),
        ]
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="answer",
            status="answered",
            claims=[],
            citations=citations,
            confidence_band="high",
            warnings=[],
        )
        metric = GroundGraphCitationMetric()
        result = await metric.evaluate(response, evidence)
        assert result.score == 0.0
        assert "cited IDs not in evidence" in (result.reason or "")


class TestGroundGraphNoHallucinationMetric:
    """Tests for hallucination metric."""

    @pytest.mark.asyncio
    async def test_supported_claims(self) -> None:
        eid = uuid4()
        evidence = [
            Evidence(
                evidence_id=eid,
                source_id=uuid4(),
                content="chunk",
                retrieval_method="vector",
            ),
        ]
        claims = [
            AnswerClaim(
                claim_id=uuid4(),
                text="supported claim",
                factual=True,
                evidence_ids=[eid],
                support_status="supported",
            ),
        ]
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="answer",
            status="answered",
            claims=claims,
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        metric = GroundGraphNoHallucinationMetric()
        result = await metric.evaluate(response, evidence)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_unsupported_claims(self) -> None:
        evidence = [
            Evidence(
                evidence_id=uuid4(),
                source_id=uuid4(),
                content="chunk",
                retrieval_method="vector",
            ),
        ]
        claims = [
            AnswerClaim(
                claim_id=uuid4(),
                text="unsupported claim",
                factual=True,
                support_status="unsupported",
            ),
        ]
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="answer",
            status="answered",
            claims=claims,
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        metric = GroundGraphNoHallucinationMetric()
        result = await metric.evaluate(response, evidence)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_evidence(self) -> None:
        response = QueryResponse(
            execution_run_id=uuid4(),
            answer="answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        metric = GroundGraphNoHallucinationMetric()
        result = await metric.evaluate(response, [])
        assert result.score == 0.5
        assert "No evidence" in (result.reason or "")
