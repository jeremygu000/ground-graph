"""Base exception hierarchy and result/error conventions.

All application-layer errors should inherit from these so that workflows
and the API can present them with stable codes and retryability hints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):  # noqa: UP042 - intentional str+Enum for JSON serialization
    """Stable, machine-readable error codes.

    Codes are part of the public API contract. Do not rename or remove
    codes without an ADR and a deprecation period.

    Note: inherits from both str and Enum to remain serializable as a
    plain string in JSON and headers while keeping enum semantics.
    """

    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    CONFLICT = "conflict"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    STRUCTURED_OUTPUT_INVALID = "structured_output_invalid"
    EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    ONTOLOGY_VIOLATION = "ontology_violation"
    AUTHORIZATION_LEAK = "authorization_leak"
    INTERNAL_ERROR = "internal_error"


class GraphRAGError(Exception):
    """Base class for all application errors."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    retryable: bool = False
    safe_message: str = "An internal error occurred."

    def __init__(
        self,
        message: str = "",
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message or self.safe_message)
        self.message = message or self.safe_message
        self.details: dict[str, Any] = dict(details or {})
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.safe_message,
            "retryable": self.retryable,
            "details": self.details,
        }


class ConfigurationError(GraphRAGError):
    code = ErrorCode.CONFIGURATION_ERROR
    safe_message = "Service is not configured correctly."
    retryable = False


class ValidationError(GraphRAGError):
    code = ErrorCode.VALIDATION_ERROR
    safe_message = "Request input is invalid."
    retryable = False


class NotFoundError(GraphRAGError):
    code = ErrorCode.NOT_FOUND
    safe_message = "The requested resource was not found."
    retryable = False


class ConflictError(GraphRAGError):
    code = ErrorCode.CONFLICT
    safe_message = "The request conflicts with current state."
    retryable = False


class UnauthorizedError(GraphRAGError):
    code = ErrorCode.UNAUTHORIZED
    safe_message = "Authentication is required."
    retryable = False


class ForbiddenError(GraphRAGError):
    code = ErrorCode.FORBIDDEN
    safe_message = "You are not allowed to perform this action."
    retryable = False


class UpstreamTimeoutError(GraphRAGError):
    code = ErrorCode.UPSTREAM_TIMEOUT
    safe_message = "An upstream dependency timed out."
    retryable = True


class UpstreamUnavailableError(GraphRAGError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    safe_message = "An upstream dependency is unavailable."
    retryable = True


class StructuredOutputInvalidError(GraphRAGError):
    code = ErrorCode.STRUCTURED_OUTPUT_INVALID
    safe_message = "Model output could not be parsed into a structured response."
    retryable = False


class EmbeddingDimensionMismatchError(GraphRAGError):
    code = ErrorCode.EMBEDDING_DIMENSION_MISMATCH
    safe_message = "Embedding dimension does not match the configured index."
    retryable = False


class InsufficientEvidenceError(GraphRAGError):
    code = ErrorCode.INSUFFICIENT_EVIDENCE
    safe_message = "There is not enough supported evidence to answer."
    retryable = False


class ConflictingEvidenceError(GraphRAGError):
    code = ErrorCode.CONFLICTING_EVIDENCE
    safe_message = "Sources conflict; the answer cannot be safely produced."
    retryable = False


class OntologyViolationError(GraphRAGError):
    code = ErrorCode.ONTOLOGY_VIOLATION
    safe_message = "The fact violates ontology constraints."
    retryable = False


class AuthorizationLeakError(GraphRAGError):
    """Raised when a code path is observed exposing unauthorized data.

    This is a critical safety error and should fail the run immediately.
    """

    code = ErrorCode.AUTHORIZATION_LEAK
    safe_message = "An authorization boundary was violated."
    retryable = False
