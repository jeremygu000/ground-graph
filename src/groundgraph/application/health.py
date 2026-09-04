"""Health check application services and ports."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class HealthReasonCode(StrEnum):
    """Safe, machine-readable readiness reason codes."""

    OK = "ok"
    TIMEOUT = "timeout"
    UNHEALTHY = "unhealthy"
    ERROR = "error"


@dataclass(slots=True, frozen=True)
class DependencyHealth:
    name: str
    healthy: bool
    reason_code: HealthReasonCode = HealthReasonCode.OK
    details: str | None = None


@runtime_checkable
class HealthChecker(Protocol):
    name: str

    async def check(self) -> DependencyHealth:
        """Return health status for a dependency."""
        ...


@dataclass(slots=True)
class HealthService:
    """Run dependency probes concurrently with bounded timeout."""

    checkers: Mapping[str, HealthChecker]
    timeout_seconds: float = 2.0

    async def check_all(self) -> list[DependencyHealth]:
        return await asyncio.gather(
            *(self._probe(name, checker) for name, checker in self.checkers.items())
        )

    async def _probe(self, name: str, checker: HealthChecker) -> DependencyHealth:
        try:
            return await asyncio.wait_for(checker.check(), timeout=self.timeout_seconds)
        except TimeoutError:
            return DependencyHealth(
                name=name,
                healthy=False,
                reason_code=HealthReasonCode.TIMEOUT,
                details="dependency timed out",
            )
        except Exception:
            return DependencyHealth(
                name=name,
                healthy=False,
                reason_code=HealthReasonCode.ERROR,
                details="dependency probe failed",
            )


def readiness_healthy(results: list[DependencyHealth]) -> bool:
    return all(result.healthy for result in results)


def readiness_status(results: list[DependencyHealth]) -> str:
    return "ok" if readiness_healthy(results) else "degraded"


def readiness_http_status(results: list[DependencyHealth]) -> int:
    return 200 if readiness_healthy(results) else 503
