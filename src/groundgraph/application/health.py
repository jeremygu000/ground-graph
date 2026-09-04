"""Health check application services and ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class DependencyHealth:
    name: str
    healthy: bool
    details: str | None = None


@runtime_checkable
class HealthChecker(Protocol):
    async def check(self) -> DependencyHealth:
        """Return health status for a dependency."""
        ...
