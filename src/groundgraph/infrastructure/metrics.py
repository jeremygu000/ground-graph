"""Prometheus metrics for the API boundary."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUEST_COUNT = Counter(
    "groundgraph_http_requests_total",
    "Total HTTP requests observed by the API boundary.",
    ("method", "route", "status_code"),
)

HTTP_REQUEST_ERRORS = Counter(
    "groundgraph_http_request_errors_total",
    "Total HTTP request failures observed by the API boundary.",
    ("method", "route", "exception_type"),
)

HTTP_REQUEST_DURATION = Histogram(
    "groundgraph_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
)

READINESS_DEPENDENCY_HEALTH = Gauge(
    "groundgraph_readiness_dependency_healthy",
    "Whether a required readiness dependency is healthy (1) or unhealthy (0).",
    ("dependency",),
)
