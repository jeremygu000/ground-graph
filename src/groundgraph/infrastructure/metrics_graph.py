"""Graph health metrics for M5."""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry.metrics import Counter, Histogram, Meter


@dataclass
class GraphHealthMetrics:
    projection_lag_ms: Histogram
    resolution_queue_depth: Histogram
    provenance_coverage_pct: Histogram
    conflict_count: Counter
    entity_count: Histogram
    fact_count: Histogram


def init_graph_health_metrics(meter: Meter) -> GraphHealthMetrics:
    """Create graph health instruments from *meter*."""
    return GraphHealthMetrics(
        projection_lag_ms=meter.create_histogram(
            "groundgraph.graph.projection.lag_ms",
            description="Lag between outbox event creation and Neo4j projection in milliseconds.",
            unit="ms",
        ),
        resolution_queue_depth=meter.create_histogram(
            "groundgraph.graph.resolution.queue_depth",
            description="Number of entities awaiting resolution.",
        ),
        provenance_coverage_pct=meter.create_histogram(
            "groundgraph.graph.provenance.coverage_pct",
            description="Percentage of facts with provenance coverage.",
        ),
        conflict_count=meter.create_counter(
            "groundgraph.graph.conflicts",
            description="Number of graph conflicts detected.",
        ),
        entity_count=meter.create_histogram(
            "groundgraph.graph.entities.count",
            description="Total number of entities in the graph.",
        ),
        fact_count=meter.create_histogram(
            "groundgraph.graph.facts.count",
            description="Total number of facts in the graph.",
        ),
    )
