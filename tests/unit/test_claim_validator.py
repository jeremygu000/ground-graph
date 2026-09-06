"""Unit tests for claim validator."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.application.answering.claim_validator import DeterministicClaimValidator
from groundgraph.domain.retrieval import AnswerClaim, Evidence


@pytest.mark.asyncio
async def test_validate_supported_claim_with_valid_evidence() -> None:
    """A claim with evidence that exists in evidence_map stays supported."""
    validator = DeterministicClaimValidator()
    evidence_id = uuid4()
    claim_id = uuid4()
    evidence = Evidence(
        evidence_id=evidence_id,
        source_id=uuid4(),
        content="Test content",
        retrieval_method="vector",
        allowed_principals=["user1"],
    )
    claim = AnswerClaim(
        claim_id=claim_id,
        text="PostgreSQL is a database.",
        factual=True,
        evidence_ids=[evidence_id],
        support_status="supported",
    )
    response = type(
        "Response",
        (),
        {
            "execution_run_id": uuid4(),
            "answer": "PostgreSQL is a database.",
            "status": "answered",
            "claims": [claim],
            "citations": [],
            "confidence_band": "high",
            "warnings": [],
        },
    )()

    result = await validator.validate(response, [evidence])

    assert len(result.claims) == 1
    assert result.claims[0].support_status == "supported"


@pytest.mark.asyncio
async def test_validate_unsupported_claim_unchanged() -> None:
    """An unsupported claim stays unsupported."""
    validator = DeterministicClaimValidator()
    claim_id = uuid4()
    claim = AnswerClaim(
        claim_id=claim_id,
        text="Ghost in the machine.",
        factual=True,
        evidence_ids=[],
        support_status="unsupported",
    )
    response = type(
        "Response",
        (),
        {
            "execution_run_id": uuid4(),
            "answer": None,
            "status": "insufficient_evidence",
            "claims": [claim],
            "citations": [],
            "confidence_band": "low",
            "warnings": [],
        },
    )()

    result = await validator.validate(response, [])

    assert len(result.claims) == 1
    assert result.claims[0].support_status == "unsupported"


@pytest.mark.asyncio
async def test_validate_downgrades_to_unsupported_when_all_evidence_missing() -> None:
    """A supported claim whose all evidence_ids are missing is downgraded to unsupported."""
    validator = DeterministicClaimValidator()
    missing_id = uuid4()
    claim_id = uuid4()
    claim = AnswerClaim(
        claim_id=claim_id,
        text="Some claim.",
        factual=True,
        evidence_ids=[missing_id],
        support_status="supported",
    )
    response = type(
        "Response",
        (),
        {
            "execution_run_id": uuid4(),
            "answer": "Some claim.",
            "status": "answered",
            "claims": [claim],
            "citations": [],
            "confidence_band": "medium",
            "warnings": [],
        },
    )()

    result = await validator.validate(response, [])

    assert len(result.claims) == 1
    assert result.claims[0].support_status == "unsupported"


@pytest.mark.asyncio
async def test_validate_partially_supported_preserves_some_evidence() -> None:
    """A partially supported claim with some evidence present keeps that evidence."""
    validator = DeterministicClaimValidator()
    present_id = uuid4()
    missing_id = uuid4()
    claim_id = uuid4()
    evidence = Evidence(
        evidence_id=present_id,
        source_id=uuid4(),
        content="Test content",
        retrieval_method="vector",
        allowed_principals=["user1"],
    )
    claim = AnswerClaim(
        claim_id=claim_id,
        text="Some claim.",
        factual=True,
        evidence_ids=[present_id, missing_id],
        support_status="supported",
    )
    response = type(
        "Response",
        (),
        {
            "execution_run_id": uuid4(),
            "answer": "Some claim.",
            "status": "answered",
            "claims": [claim],
            "citations": [],
            "confidence_band": "medium",
            "warnings": [],
        },
    )()

    result = await validator.validate(response, [evidence])

    assert len(result.claims) == 1
    assert result.claims[0].support_status == "partially_supported"
    assert result.claims[0].evidence_ids == [present_id]
