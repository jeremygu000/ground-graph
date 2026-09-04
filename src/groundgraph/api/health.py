"""Health routes for the API."""

from __future__ import annotations

from collections.abc import MutableMapping

from fastapi import APIRouter, Request, Response
from opentelemetry.metrics import Observation
from pydantic import BaseModel, Field

from groundgraph.application.health import (
    HealthService,
    readiness_http_status,
    readiness_status,
)

router = APIRouter(tags=["health"])


class PublicDependencyHealth(BaseModel):
    """Readiness detail safe to disclose to an unauthenticated caller."""

    name: str
    healthy: bool
    reason_code: str


class HealthResponse(BaseModel):
    status: str
    dependencies: list[PublicDependencyHealth] = Field(default_factory=list)


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


def _init_readiness_gauge(
    meter: object,
    state: MutableMapping[str, int],
) -> None:
    def callback(_options: object) -> list[Observation]:
        return [Observation(float(value), {"dependency": dep}) for dep, value in state.items()]

    meter.create_observable_gauge(
        "groundgraph.readiness.dependency.healthy",
        description="Current health state of each readiness dependency (1=healthy, 0=unhealthy).",
        callbacks=[callback],
    )


@router.get("/health/ready", response_model=HealthResponse)
async def ready(
    request: Request,
    response: Response,
) -> HealthResponse:
    health_service: HealthService = request.app.state.health_service
    meter = getattr(request.app.state, "meter", None)

    if not hasattr(request.app.state, "_readiness_gauge_state"):
        request.app.state._readiness_gauge_state: dict[str, int] = {}
        if meter is not None:
            _init_readiness_gauge(meter, request.app.state._readiness_gauge_state)

    dependencies = await health_service.check_all()
    state: dict[str, int] = request.app.state._readiness_gauge_state
    for dependency in dependencies:
        state[dependency.name] = 1 if dependency.healthy else 0

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
