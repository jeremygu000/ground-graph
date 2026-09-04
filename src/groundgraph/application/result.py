"""Result envelope conventions for application services.

Use ``Ok`` and ``Err`` for fallible operations where callers want explicit
error handling without exceptions. Critical errors (authorization leaks,
internal faults) should still raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from groundgraph.application.errors import ErrorCode, GraphRAGError

T = TypeVar("T")
E = TypeVar("E", bound=GraphRAGError)


@dataclass(slots=True, frozen=True)
class Ok[T]:
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False


@dataclass(slots=True, frozen=True)
class Err[E: GraphRAGError]:
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True


Result = Ok[T] | Err[E]


def to_error_dict(code: ErrorCode, message: str, retryable: bool) -> dict[str, str | bool]:
    return {"code": code.value, "message": message, "retryable": retryable}
