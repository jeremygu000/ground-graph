"""Health routes for the API."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from fastapi import APIRouter
from pydantic import BaseModel

from groundgraph.application.health import DependencyHealth

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    dependencies: list[DependencyHealth] = []


async def _probe_dependency(
    name: str,
    probe: Callable[[], Awaitable[DependencyHealth]],
) -> DependencyHealth:
    try:
        return await asyncio.wait_for(probe(), timeout=2.0)
    except TimeoutError:
        return DependencyHealth(name=name, healthy=False, details="timeout")
    except Exception as exc:  # pragma: no cover - exercised in integration later
        return DependencyHealth(name=name, healthy=False, details=str(exc))


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    return HealthResponse(status="ok")
