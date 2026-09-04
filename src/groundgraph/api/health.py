"""Health routes for the API."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from groundgraph.application.health import (
    HealthService,
    readiness_http_status,
    readiness_status,
)
from groundgraph.infrastructure.metrics import READINESS_DEPENDENCY_HEALTH

router = APIRouter(tags=["health"])


class PublicDependencyHealth(BaseModel):
    """Readiness detail safe to disclose to an unauthenticated caller."""

    name: str
    healthy: bool
    reason_code: str


class HealthResponse(BaseModel):
    status: str
    dependencies: list[PublicDependencyHealth] = []


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def ready(
    request: Request,
    response: Response,
) -> HealthResponse:
    health_service: HealthService = request.app.state.health_service
    dependencies = await health_service.check_all()
    for dependency in dependencies:
        READINESS_DEPENDENCY_HEALTH.labels(dependency.name).set(int(dependency.healthy))
    response.status_code = readiness_http_status(dependencies)
    public_dependencies = [
        PublicDependencyHealth(
            name=dependency.name,
            healthy=dependency.healthy,
            reason_code=dependency.reason_code,
        )
        for dependency in dependencies
    ]
    return HealthResponse(status=readiness_status(dependencies), dependencies=public_dependencies)
