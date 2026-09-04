"""Health check application services and ports."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


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


@runtime_checkable
class HealthChecker(Protocol):
    name: str

    async def check(self) -> DependencyHealth:
        """Return health status for a dependency."""
        ...


@dataclass(slots=True)
class HealthService:
    """Run dependency probes concurrently with bounded timeout and OTel spans."""

    checkers: Mapping[str, HealthChecker]
    timeout_seconds: float = 2.0
    tracer: Tracer | None = None

    async def check_all(self) -> list[DependencyHealth]:
        return await asyncio.gather(
            *(self._probe(name, checker) for name, checker in self.checkers.items())
        )

    async def _probe(self, name: str, checker: HealthChecker) -> DependencyHealth:
        from opentelemetry import trace  # noqa: PLC0415 - domain layer avoids OTel imports

        tracer = self.tracer or trace.get_tracer(__name__)
        operation = "check"
        with tracer.start_as_current_span(
            f"healthcheck.{name}",
            kind=trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("dependency.name", name)
            span.set_attribute("dependency.operation", operation)
            start = asyncio.get_event_loop().time()
            try:
                result = await asyncio.wait_for(checker.check(), timeout=self.timeout_seconds)
                duration_ms = (asyncio.get_event_loop().time() - start) * 1000
                span.set_attribute(
                    "dependency.result", "healthy" if result.healthy else "unhealthy"
                )
                span.set_attribute("dependency.reason_code", result.reason_code.value)
                span.set_attribute("duration_ms", round(duration_ms, 3))
                if not result.healthy:
                    span.set_attribute("error.type", result.reason_code.value)
                return result  # noqa: TRY300 - conditional error.type requires pre-return setup
            except TimeoutError:
                duration_ms = (asyncio.get_event_loop().time() - start) * 1000
                span.set_attribute("dependency.result", "timeout")
                span.set_attribute("error.type", "TimeoutError")
                span.set_attribute("duration_ms", round(duration_ms, 3))
                return DependencyHealth(
                    name=name,
                    healthy=False,
                    reason_code=HealthReasonCode.TIMEOUT,
                )
            except Exception as exc:
                duration_ms = (asyncio.get_event_loop().time() - start) * 1000
                span.set_attribute("dependency.result", "error")
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("duration_ms", round(duration_ms, 3))
                return DependencyHealth(
                    name=name,
                    healthy=False,
                    reason_code=HealthReasonCode.ERROR,
                )


def readiness_healthy(results: list[DependencyHealth]) -> bool:
    return all(result.healthy for result in results)


def readiness_status(results: list[DependencyHealth]) -> str:
    return "ok" if readiness_healthy(results) else "degraded"


def readiness_http_status(results: list[DependencyHealth]) -> int:
    return 200 if readiness_healthy(results) else 503
