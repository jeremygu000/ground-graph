"""M1.3 readiness health-service tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from groundgraph.application.health import (
    DependencyHealth,
    HealthReasonCode,
    HealthService,
    readiness_healthy,
    readiness_status,
)


@dataclass(slots=True)
class _Checker:
    name: str
    healthy: bool = True
    delay: float = 0.0
    reason_code: HealthReasonCode = HealthReasonCode.OK

    async def check(self) -> DependencyHealth:
        if self.delay:
            await asyncio.sleep(self.delay)
        return DependencyHealth(
            name=self.name,
            healthy=self.healthy,
            reason_code=self.reason_code if not self.healthy else HealthReasonCode.OK,
        )


@pytest.mark.anyio
async def test_health_service_runs_checkers_concurrently() -> None:
    service = HealthService(
        checkers={
            "postgres": _Checker("postgres", delay=0.05),
            "neo4j": _Checker("neo4j", delay=0.05),
            "minio": _Checker("minio", delay=0.05),
        },
        timeout_seconds=1.0,
    )
    started = asyncio.get_event_loop().time()
    results = await service.check_all()
    elapsed = asyncio.get_event_loop().time() - started
    assert len(results) == 3
    assert elapsed < 0.12
    assert readiness_healthy(results) is True
    assert readiness_status(results) == "ok"


@pytest.mark.anyio
async def test_health_service_timeout_returns_safe_reason() -> None:
    class _SlowChecker:
        name = "postgres"

        async def check(self) -> DependencyHealth:
            await asyncio.sleep(0.2)
            return DependencyHealth(name=self.name, healthy=True)

    service = HealthService(checkers={"postgres": _SlowChecker()}, timeout_seconds=0.01)
    results = await service.check_all()
    assert results[0].healthy is False
    assert results[0].reason_code is HealthReasonCode.TIMEOUT
    assert results[0].details is None


@pytest.mark.anyio
async def test_health_service_unhealthy_returns_safe_reason() -> None:
    service = HealthService(
        checkers={
            "minio": _Checker(
                "minio",
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
            )
        }
    )
    results = await service.check_all()
    assert results[0].healthy is False
    assert results[0].reason_code is HealthReasonCode.UNHEALTHY
    assert readiness_healthy(results) is False
    assert readiness_status(results) == "degraded"
