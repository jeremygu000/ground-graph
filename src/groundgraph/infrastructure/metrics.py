"""OTel metrics for the API boundary.

All instruments are created from the app-local MeterProvider so that test code
can inject an InMemoryMetricReader and assert on recorded values without any
network export.
"""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.metrics import Counter, Histogram, Meter


@dataclass
class AppMetrics:
    http_request_count: Counter
    http_request_errors: Counter
    http_request_duration: Histogram


def init_app_metrics(meter: Meter) -> AppMetrics:
    """Create API-bound instruments from *meter*."""
    return AppMetrics(
        http_request_count=meter.create_counter(
            "groundgraph.http.requests",
            description="Total HTTP requests observed by the API boundary.",
        ),
        http_request_errors=meter.create_counter(
            "groundgraph.http.request.errors",
            description="Total HTTP request failures (5xx).",
        ),
        http_request_duration=meter.create_histogram(
            "groundgraph.http.request.duration",
            description="HTTP request duration in seconds.",
            unit="s",
        ),
    )
