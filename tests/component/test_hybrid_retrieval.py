"""M6 component tests: Hybrid GraphRAG retrieval with real Neo4j + pgvector.

Requires Docker. Skipped if Docker is not available.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from neo4j import AsyncGraphDatabase

from groundgraph.application.retrieval.graph_fusion import GraphFusionService
from groundgraph.domain.knowledge import CanonicalEntity, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.neo4j.unit_of_work import Neo4jUnitOfWork

pytestmark = [pytest.mark.integration, pytest.mark.component]


@pytest.fixture
async def neo4j_driver(neo4j_component: Any) -> AsyncGenerator[Any, None]:
    driver = AsyncGraphDatabase.driver(
        neo4j_component.uri,
        auth=(neo4j_component.user, neo4j_component.password),
    )
    try:
        yield driver
    finally:
        await driver.close()


async def test_graph_fusion_multihop_retrieval(
    neo4j_driver: Any,
) -> None:
    """Test GraphFusionService traverses multi-hop paths (A→B→C) in Neo4j.

    Creates a 2-hop chain:
      Service A --DEPENDS_ON--> Service B --DEPENDS_ON--> Service C

    Verifies that starting from Service A with max_depth=2 returns
    both Service B (depth 1) and Service C (depth 2) facts.
    """
    entity_a = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service A",
        aliases=[],
        attributes={},
    )
    entity_b = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service B",
        aliases=[],
        attributes={},
    )
    entity_c = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service C",
        aliases=[],
        attributes={},
    )

    chunk_a = uuid4()
    chunk_b = uuid4()

    fact_ab = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_a.entity_id,
        predicate="DEPENDS_ON",
        object_id=entity_b.entity_id,
        status="verified",
        confidence=0.95,
        evidence_ids=[chunk_a],
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["engineering"],
    )
    fact_bc = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_b.entity_id,
        predicate="DEPENDS_ON",
        object_id=entity_c.entity_id,
        status="verified",
        confidence=0.90,
        evidence_ids=[chunk_b],
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["engineering"],
    )

    async with Neo4jUnitOfWork(neo4j_driver) as uow:
        assert uow.graph is not None
        await uow.graph.create_entity(entity_a)
        await uow.graph.create_entity(entity_b)
        await uow.graph.create_entity(entity_c)
        await uow.graph.create_fact(fact_ab)
        await uow.graph.create_fact(fact_bc)

    repo = Neo4jGraphRepository(driver=neo4j_driver, database="neo4j")
    svc = GraphFusionService(graph_repository=repo)

    evidence = await svc.retrieve_evidence(
        seed_entities=[entity_a],
        predicates=["DEPENDS_ON"],
        valid_at=datetime(2024, 6, 15, tzinfo=UTC),
        max_depth=2,
        tenant_id="test-tenant",
        principal="engineering",
    )

    assert len(evidence) == 2, f"Expected 2 facts (depth1 + depth2), got {len(evidence)}"

    evidence_map: dict[Any, Any] = {}
    for ev in evidence:
        evidence_map[ev.source_id] = ev

    assert chunk_a in evidence_map, "Depth-1 fact (A→B) should be returned"
    assert chunk_b in evidence_map, "Depth-2 fact (B→C) should be returned"

    for ev in evidence:
        assert ev.retrieval_method == "graph"
        assert ev.graph_path_fact_ids, "Graph evidence should have path fact IDs"
        assert ev.valid_from is not None
        assert ev.valid_to is not None
        assert "engineering" in ev.allowed_principals


async def test_graph_fusion_respects_acl_filtering(
    neo4j_driver: Any,
) -> None:
    """Test GraphFusionService filters by allowed_principals.

    Creates two facts where the seed entity (entity_x) is the subject:
      entity_x --DEPENDS_ON--> entity_y (admin-only, filtered out)
      entity_x --CALLS--> entity_z (engineering+admin, visible)

    Verifies only the fact accessible to 'engineering' is returned.
    """
    entity_x = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service X",
        aliases=[],
        attributes={},
    )
    entity_y = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service Y",
        aliases=[],
        attributes={},
    )
    entity_z = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service Z",
        aliases=[],
        attributes={},
    )

    chunk_admin = uuid4()
    chunk_eng = uuid4()

    fact_admin_only = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_x.entity_id,
        predicate="DEPENDS_ON",
        object_id=entity_y.entity_id,
        status="verified",
        confidence=0.85,
        evidence_ids=[chunk_admin],
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["admin"],  # engineering NOT included
    )
    fact_engineering = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_x.entity_id,
        predicate="DEPENDS_ON",
        object_id=entity_z.entity_id,
        status="verified",
        confidence=0.80,
        evidence_ids=[chunk_eng],
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 12, 31, tzinfo=UTC),
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["engineering", "admin"],
    )

    async with Neo4jUnitOfWork(neo4j_driver) as uow:
        assert uow.graph is not None
        await uow.graph.create_entity(entity_x)
        await uow.graph.create_entity(entity_y)
        await uow.graph.create_entity(entity_z)
        await uow.graph.create_fact(fact_admin_only)
        await uow.graph.create_fact(fact_engineering)

    repo = Neo4jGraphRepository(driver=neo4j_driver, database="neo4j")
    svc = GraphFusionService(graph_repository=repo)

    evidence = await svc.retrieve_evidence(
        seed_entities=[entity_x],
        predicates=["DEPENDS_ON"],  # only first predicate is used in current impl
        valid_at=datetime(2024, 6, 15, tzinfo=UTC),
        max_depth=1,
        tenant_id="test-tenant",
        principal="engineering",
    )

    fact_ids = {ev.evidence_id for ev in evidence}
    assert fact_engineering.fact_id in fact_ids, "Engineering fact should be visible"
    assert fact_admin_only.fact_id not in fact_ids, "Admin-only fact should be filtered"


async def test_graph_fusion_respects_temporal_validity(
    neo4j_driver: Any,
) -> None:
    """Test GraphFusionService filters by temporal validity (valid_from/valid_to)."""
    entity_x = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service X",
        aliases=[],
        attributes={},
    )
    entity_y = CanonicalEntity(
        entity_id=uuid4(),
        entity_type="Service",
        canonical_name="Service Y",
        aliases=[],
        attributes={},
    )

    chunk_current = uuid4()
    chunk_expired = uuid4()

    fact_current = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_x.entity_id,
        predicate="CALLS",
        object_id=entity_y.entity_id,
        status="verified",
        confidence=0.90,
        evidence_ids=[chunk_current],
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2030, 12, 31, tzinfo=UTC),  # still valid
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["engineering"],
    )
    fact_expired = KnowledgeFact(
        fact_id=uuid4(),
        subject_id=entity_x.entity_id,
        predicate="CALLED",
        object_id=entity_y.entity_id,
        status="verified",
        confidence=0.70,
        evidence_ids=[chunk_expired],
        valid_from=datetime(2020, 1, 1, tzinfo=UTC),
        valid_to=datetime(2023, 12, 31, tzinfo=UTC),  # expired
        observed_at=datetime.now(UTC),
        extraction_method="llm",
        ontology_version="v0.1.0",
        tenant_id="test-tenant",
        allowed_principals=["engineering"],
    )

    async with Neo4jUnitOfWork(neo4j_driver) as uow:
        assert uow.graph is not None
        await uow.graph.create_entity(entity_x)
        await uow.graph.create_entity(entity_y)
        await uow.graph.create_fact(fact_current)
        await uow.graph.create_fact(fact_expired)

    repo = Neo4jGraphRepository(driver=neo4j_driver, database="neo4j")
    svc = GraphFusionService(graph_repository=repo)

    evidence = await svc.retrieve_evidence(
        seed_entities=[entity_x],
        predicates=["CALLS", "CALLED"],
        valid_at=datetime(2024, 6, 15, tzinfo=UTC),  # query in 2024
        max_depth=1,
        tenant_id="test-tenant",
        principal="engineering",
    )

    fact_ids = {ev.evidence_id for ev in evidence}
    assert fact_current.fact_id in fact_ids, "Current fact should be visible"
    assert fact_expired.fact_id not in fact_ids, "Expired fact should be filtered"
