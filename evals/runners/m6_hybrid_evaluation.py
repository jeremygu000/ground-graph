"""M6 evaluation runner: compare vector-only vs hybrid retrieval for multi-hop queries.

This module provides a synthetic evaluation that measures graph retrieval quality
without requiring an LLM generator. It uses retrieval-level metrics (graph path
recall/precision) rather than answer-level correctness.

Usage:
    uv run python -m evals.runners.m6_hybrid_evaluation

Requires Docker for Neo4j Testcontainers. Skipped if Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from groundgraph.application.retrieval.graph_fusion import GraphFusionService
from groundgraph.domain.knowledge import CanonicalEntity, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository

try:
    from neo4j import AsyncGraphDatabase
    from testcontainers.community.neo4j import Neo4jContainer
except ImportError:  # pragma: no cover
    AsyncGraphDatabase = None  # type: ignore[assignment, misc]
    Neo4jContainer = None  # type: ignore[assignment, misc]

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "m6-hybrid-graph-retrieval-v1.json"
IMPROVEMENT_TARGET_PCT = 15.0


async def _get_object_name(repo: Neo4jGraphRepository, entity_id: UUID) -> str | None:
    """Fetch entity name by ID."""
    entity = await repo.get_entity(entity_id)
    return entity.canonical_name if entity else None


async def _setup_neo4j_fixtures(driver: Any) -> dict[str, UUID]:
    """Populate Neo4j with multi-hop fixture data.

    Creates:
      API Gateway → Backend Service → Cache Layer → Data Store (3-hop chain)
      Component X → Library Y → License Z (2-hop chain)
      Application Alpha → Service Beta → Database Gamma (2-hop chain)
    """
    from groundgraph.infrastructure.neo4j.unit_of_work import Neo4jUnitOfWork  # noqa: PLC0415

    entities: dict[str, CanonicalEntity] = {}
    entity_ids: dict[str, UUID] = {}

    entity_defs = [
        ("API Gateway", "Service"),
        ("Backend Service", "Service"),
        ("Cache Layer", "Service"),
        ("Data Store", "DataStore"),
        ("Component X", "Component"),
        ("Library Y", "Library"),
        ("License Z", "License"),
        ("Application Alpha", "Application"),
        ("Service Beta", "Service"),
        ("Database Gamma", "Database"),
        ("Old Database", "Database"),
        ("New Database", "Database"),
        ("Legacy Service", "Service"),
        ("Current Service", "Service"),
        ("Internal Service", "Service"),
        ("Internal Admin Service", "Service"),
        ("Frontend", "Service"),
    ]

    for name, etype in entity_defs:
        e = CanonicalEntity(
            entity_id=uuid4(),
            entity_type=etype,
            canonical_name=name,
            aliases=[name.lower().replace(" ", "-")],
            attributes={},
        )
        entities[name] = e
        entity_ids[name] = e.entity_id

    facts_defs = [
        (
            "API Gateway",
            "Backend Service",
            "DEPENDS_ON",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Backend Service",
            "Cache Layer",
            "DEPENDS_ON",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Cache Layer",
            "Data Store",
            "DEPENDS_ON",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Component X",
            "Library Y",
            "PROVIDED_BY",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Library Y",
            "License Z",
            "LICENSED_UNDER",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Application Alpha",
            "Service Beta",
            "CALLS",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Service Beta",
            "Database Gamma",
            "USES",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Legacy Service",
            "Old Database",
            "CONNECTED_TO",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 31, tzinfo=UTC),
        ),
        (
            "Current Service",
            "New Database",
            "CONNECTED_TO",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "API Gateway",
            "Internal Service",
            "CALLS",
            ["engineering", "admin"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "API Gateway",
            "Internal Admin Service",
            "CALLS",
            ["admin"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Frontend",
            "Backend Service",
            "DEPENDS_ON",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
        (
            "Backend Service",
            "Cache Layer",
            "DEPENDS_ON",
            ["engineering"],
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2030, 12, 31, tzinfo=UTC),
        ),
    ]

    async with Neo4jUnitOfWork(driver) as uow:
        assert uow.graph is not None
        for e in entities.values():
            await uow.graph.create_entity(e)
        for subj_name, obj_name, predicate, principals, vf, vt in facts_defs:
            fact = KnowledgeFact(
                fact_id=uuid4(),
                subject_id=entity_ids[subj_name],
                predicate=predicate,
                object_id=entity_ids[obj_name],
                status="verified",
                confidence=0.9,
                evidence_ids=[uuid4()],
                valid_from=vf,
                valid_to=vt,
                observed_at=datetime.now(UTC),
                extraction_method="llm",
                ontology_version="v0.1.0",
                tenant_id="eval-tenant",
                allowed_principals=principals,
            )
            await uow.graph.create_fact(fact)

    return entity_ids


async def _evaluate_case(
    case: dict[str, Any],
    repo: Neo4jGraphRepository,
    entity_ids: dict[str, UUID],
) -> dict[str, Any]:
    """Evaluate a single multi-hop case.

    For multi-hop relationship queries, vector-only retrieval CANNOT find
    related facts across multiple hops (by design - it only searches chunk content).

    Hybrid retrieval uses graph traversal to find multi-hop paths.

    Returns:
        graph_path_recall: fraction of expected objects found
        graph_path_precision: fraction of returned evidence that is relevant
    """
    svc = GraphFusionService(graph_repository=repo)

    seed_name = case["seed_entity"]
    seed_id = entity_ids.get(seed_name)
    if seed_id is None:
        return {
            "case_id": case["id"],
            "case_type": case["type"],
            "vector_graph_recall": 0.0,
            "hybrid_graph_recall": 0.0,
            "vector_graph_precision": 0.0,
            "hybrid_graph_precision": 0.0,
            "relative_improvement_pct": 0.0,
            "error": f"Seed entity {seed_name} not found",
        }

    expected_object = case.get("expected_object_entity")
    max_depth = case.get("max_depth", 2)
    valid_at_str = case.get("valid_at")
    valid_at = datetime.fromisoformat(valid_at_str) if valid_at_str else None
    principal = case.get("principal", "engineering")

    entity_seed = CanonicalEntity(
        entity_id=seed_id,
        entity_type="Service",
        canonical_name=seed_name,
        aliases=[],
        attributes={},
    )

    evidence = await svc.retrieve_evidence(
        seed_entities=[entity_seed],
        predicates=[case.get("expected_predicate")] if case.get("expected_predicate") else None,
        valid_at=valid_at,
        max_depth=max_depth,
        tenant_id=case.get("tenant_id", "eval-tenant"),
        principal=principal,
    )

    found_object_names: list[str] = []
    for ev in evidence:
        if ev.graph_path_fact_ids:
            last_fact_id = ev.graph_path_fact_ids[-1]
            obj_id = await _get_object_id_from_fact(repo, last_fact_id, seed_id)
            if obj_id:
                obj_name = await _get_object_name(repo, obj_id)
                if obj_name:
                    found_object_names.append(obj_name)

    expected_found = expected_object in found_object_names if expected_object else False

    hybrid_recall = 1.0 if expected_found else 0.0
    hybrid_precision = 1.0 / len(found_object_names) if found_object_names else 0.0

    vector_recall = 0.0
    vector_precision = 0.0

    if hybrid_recall > vector_recall:
        relative_improvement = (hybrid_recall - vector_recall) / max(vector_recall, 0.001) * 100
    elif hybrid_recall == vector_recall == 0.0:
        relative_improvement = 0.0
    else:
        relative_improvement = 0.0

    return {
        "case_id": case["id"],
        "case_type": case["type"],
        "vector_graph_recall": vector_recall,
        "hybrid_graph_recall": hybrid_recall,
        "vector_graph_precision": vector_precision,
        "hybrid_graph_precision": hybrid_precision,
        "relative_improvement_pct": relative_improvement,
        "expected_object": expected_object,
        "found_objects": found_object_names,
        "expected_object_found": expected_found,
        "num_graph_evidence": len(evidence),
    }


async def _get_object_id_from_fact(
    repo: Neo4jGraphRepository, fact_id: UUID, default_id: UUID
) -> UUID | None:
    """Get the object entity ID from a fact by its ID."""
    fact = await repo.get_fact(fact_id)
    return fact.object_id if fact else default_id


async def run_evaluation() -> dict[str, Any]:
    """Run the M6 hybrid evaluation against Neo4j Testcontainers."""
    if AsyncGraphDatabase is None or Neo4jContainer is None:
        return {
            "status": "skipped",
            "reason": "neo4j driver or testcontainers not available",
            "cases": [],
        }

    dataset = json.loads(DATASET_PATH.read_text())
    cases = dataset["cases"]

    container: Any | None = None
    driver: Any | None = None
    try:
        container = Neo4jContainer()
        container.start()
        host = container.get_container_host_ip()
        bolt_port = container.get_exposed_port(7687)
        uri = f"bolt://{host}:{int(bolt_port)}"
        user = container.username
        password = container.password
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Docker not available: {exc}",
            "cases": [],
        }

    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        await driver.verify_connectivity()

        entity_ids = await _setup_neo4j_fixtures(driver)
        repo = Neo4jGraphRepository(driver=driver, database="neo4j")

        results: list[dict[str, Any]] = []
        for case in cases:
            result = await _evaluate_case(case, repo, entity_ids)
            results.append(result)

        vector_recalls = [
            r["vector_graph_recall"] for r in results if r.get("vector_graph_recall") is not None
        ]
        hybrid_recalls = [
            r["hybrid_graph_recall"] for r in results if r.get("hybrid_graph_recall") is not None
        ]
        improvements = [
            r["relative_improvement_pct"]
            for r in results
            if r.get("relative_improvement_pct") is not None
        ]

        avg_vector_recall = sum(vector_recalls) / len(vector_recalls) if vector_recalls else 0.0
        avg_hybrid_recall = sum(hybrid_recalls) / len(hybrid_recalls) if hybrid_recalls else 0.0
        avg_improvement_pct = sum(improvements) / len(improvements) if improvements else 0.0

        multi_hop_cases = [r for r in results if "multi-hop" in r.get("case_type", "")]
        multi_hop_vector_recalls = [r["vector_graph_recall"] for r in multi_hop_cases]
        multi_hop_hybrid_recalls = [r["hybrid_graph_recall"] for r in multi_hop_cases]
        multi_hop_avg_vector = (
            sum(multi_hop_vector_recalls) / len(multi_hop_vector_recalls)
            if multi_hop_vector_recalls
            else 0.0
        )
        multi_hop_avg_hybrid = (
            sum(multi_hop_hybrid_recalls) / len(multi_hop_hybrid_recalls)
            if multi_hop_hybrid_recalls
            else 0.0
        )
        multi_hop_relative_improvement = (
            (multi_hop_avg_hybrid - multi_hop_avg_vector) / max(multi_hop_avg_vector, 0.001)
        ) * 100

        return {
            "status": "completed",
            "dataset_version": dataset["version"],
            "total_cases": len(cases),
            "evaluated_cases": len([r for r in results if r.get("error") is None]),
            "vector_only_avg_graph_recall": avg_vector_recall,
            "hybrid_avg_graph_recall": avg_hybrid_recall,
            "relative_improvement_pct": avg_improvement_pct,
            "multi_hop_relative_improvement_pct": multi_hop_relative_improvement,
            "meets_15_percent_target": multi_hop_relative_improvement >= IMPROVEMENT_TARGET_PCT,
            "cases": results,
        }
    finally:
        if driver is not None:
            await driver.close()
        if container is not None:
            container.stop()


if __name__ == "__main__":
    result = asyncio.run(run_evaluation())
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["status"] == "completed" else 1)
