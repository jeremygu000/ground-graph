"""FastAPI dependencies and request-scoped helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request


def get_request_id(request: Request) -> str | None:
    return request.headers.get("x-request-id") or request.headers.get("x-correlation-id")


@asynccontextmanager
async def lifespan_marker() -> AsyncIterator[None]:
    """Placeholder lifespan hook used by the composition root."""

    yield
