"""Tests for the application error hierarchy and result envelope."""

from __future__ import annotations

import pytest

from graphrag.application.errors import (
    AuthorizationLeakError,
    ConfigurationError,
    ConflictError,
    ConflictingEvidenceError,
    EmbeddingDimensionMismatchError,
    ErrorCode,
    ForbiddenError,
    GraphRAGError,
    InsufficientEvidenceError,
    NotFoundError,
    OntologyViolationError,
    StructuredOutputInvalidError,
    UnauthorizedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    ValidationError,
)
from graphrag.application.result import Err, Ok


@pytest.mark.parametrize(
    ("exc_cls", "expected_code"),
    [
        (ConfigurationError, ErrorCode.CONFIGURATION_ERROR),
        (ValidationError, ErrorCode.VALIDATION_ERROR),
        (NotFoundError, ErrorCode.NOT_FOUND),
        (ConflictError, ErrorCode.CONFLICT),
        (UnauthorizedError, ErrorCode.UNAUTHORIZED),
        (ForbiddenError, ErrorCode.FORBIDDEN),
        (UpstreamTimeoutError, ErrorCode.UPSTREAM_TIMEOUT),
        (UpstreamUnavailableError, ErrorCode.UPSTREAM_UNAVAILABLE),
        (StructuredOutputInvalidError, ErrorCode.STRUCTURED_OUTPUT_INVALID),
        (EmbeddingDimensionMismatchError, ErrorCode.EMBEDDING_DIMENSION_MISMATCH),
        (InsufficientEvidenceError, ErrorCode.INSUFFICIENT_EVIDENCE),
        (ConflictingEvidenceError, ErrorCode.CONFLICTING_EVIDENCE),
        (OntologyViolationError, ErrorCode.ONTOLOGY_VIOLATION),
        (AuthorizationLeakError, ErrorCode.AUTHORIZATION_LEAK),
    ],
)
def test_error_code_is_stable(exc_cls: type[GraphRAGError], expected_code: ErrorCode) -> None:
    err = exc_cls("details")
    assert err.code == expected_code
    assert err.to_dict()["code"] == expected_code.value


def test_retryable_classification() -> None:
    assert UpstreamTimeoutError("x").retryable is True
    assert UpstreamUnavailableError("x").retryable is True
    assert ValidationError("x").retryable is False
    assert AuthorizationLeakError("x").retryable is False


def test_safe_message_is_never_empty() -> None:
    for cls in (
        GraphRAGError,
        ConfigurationError,
        ValidationError,
        NotFoundError,
        ConflictError,
        UnauthorizedError,
        ForbiddenError,
        UpstreamTimeoutError,
        UpstreamUnavailableError,
        StructuredOutputInvalidError,
        EmbeddingDimensionMismatchError,
        InsufficientEvidenceError,
        ConflictingEvidenceError,
        OntologyViolationError,
        AuthorizationLeakError,
    ):
        err = cls()
        assert err.safe_message, f"{cls.__name__} must define a safe message"


def test_error_details_are_isolated() -> None:
    err = ValidationError("oops", details={"field": "x"})
    assert err.details == {"field": "x"}
    err.details["leak"] = 1
    other = ValidationError("oops")
    assert "leak" not in other.details


def test_authorization_leak_is_critical() -> None:
    err = AuthorizationLeakError("forbidden path")
    assert err.code == ErrorCode.AUTHORIZATION_LEAK
    assert err.retryable is False
    assert "authorization" in err.safe_message.lower()


def test_ok_result() -> None:
    r: Ok[int] = Ok(42)
    assert r.is_ok() is True
    assert r.is_err() is False
    assert r.value == 42


def test_err_result() -> None:
    e = ValidationError("bad")
    r: Err[ValidationError] = Err(e)
    assert r.is_ok() is False
    assert r.is_err() is True
    assert r.error.code == ErrorCode.VALIDATION_ERROR


def test_error_to_dict_shape() -> None:
    err = ValidationError("oops", details={"field": "x"})
    d = err.to_dict()
    assert set(d.keys()) == {"code", "message", "retryable", "details"}
    assert d["retryable"] is False
