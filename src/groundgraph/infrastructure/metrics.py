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


@dataclass
class RetrievalMetrics:
    embedding_duration_ms: Histogram
    embedding_tokens: Counter
    vector_retrieval_duration_ms: Histogram
    keyword_retrieval_duration_ms: Histogram
    rerank_duration_ms: Histogram
    rerank_evidence_count: Histogram
    generate_duration_ms: Histogram
    generate_tokens: Counter
    generate_cost_usd: Histogram


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


def init_retrieval_metrics(meter: Meter) -> RetrievalMetrics:
    """Create retrieval-stage instruments from *meter*."""
    return RetrievalMetrics(
        embedding_duration_ms=meter.create_histogram(
            "groundgraph.retrieval.embedding.duration",
            description="Embedding call duration in milliseconds.",
            unit="ms",
        ),
        embedding_tokens=meter.create_counter(
            "groundgraph.retrieval.embedding.tokens",
            description="Total embedding tokens consumed.",
        ),
        vector_retrieval_duration_ms=meter.create_histogram(
            "groundgraph.retrieval.vector.duration",
            description="Vector retrieval duration in milliseconds.",
            unit="ms",
        ),
        keyword_retrieval_duration_ms=meter.create_histogram(
            "groundgraph.retrieval.keyword.duration",
            description="Keyword retrieval duration in milliseconds.",
            unit="ms",
        ),
        rerank_duration_ms=meter.create_histogram(
            "groundgraph.retrieval.rerank.duration",
            description="Reranking duration in milliseconds.",
            unit="ms",
        ),
        rerank_evidence_count=meter.create_histogram(
            "groundgraph.retrieval.rerank.evidence_count",
            description="Number of evidence chunks sent to reranker.",
        ),
        generate_duration_ms=meter.create_histogram(
            "groundgraph.retrieval.generate.duration",
            description="Answer generation duration in milliseconds.",
            unit="ms",
        ),
        generate_tokens=meter.create_counter(
            "groundgraph.retrieval.generate.tokens",
            description="Total generation tokens consumed.",
        ),
        generate_cost_usd=meter.create_histogram(
            "groundgraph.retrieval.generate.cost_usd",
            description="Estimated generation cost in USD.",
            unit="USD",
        ),
    )
