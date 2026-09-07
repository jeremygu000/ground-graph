"""M6 evaluation runner: compare vector-only vs graph-only vs hybrid retrieval.

This module provides a synthetic evaluation that measures retrieval quality
across three strategies for multi-hop relationship queries:
  1. Vector-only: pgvector similarity search using the user question embedding
  2. Graph-only:  Neo4j graph traversal at full depth (max_depth=2)
  3. Hybrid:      True fusion of pgvector results + graph traversal via RRF

For these graph-structured multi-hop queries, vector-only retrieval correctly
returns little or no evidence (the relationships are in the graph, not chunks).
Graph traversal at depth > 1 finds the transitive paths.

Usage:
    uv run python -m evals.runners.m6_hybrid_evaluation

Requires Docker for Neo4j AND pgvector Testcontainers. Skipped if Docker
or either container is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from groundgraph.application.retrieval.graph_fusion import GraphFusionService, hybrid_rrf_fusion
from groundgraph.domain.knowledge import CanonicalEntity, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.neo4j.unit_of_work import Neo4jUnitOfWork


@dataclass
class _FakeRetrievedChunk:
    chunk_id: Any
    source_id: Any
    document_id: Any
    version_id: Any
    content: str
    vector_score: float | None
    keyword_score: float | None
    allowed_principals: list[str]


try:
    import asyncpg
    from neo4j import AsyncGraphDatabase
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.community.postgres import PostgresContainer
except ImportError:  # pragma: no cover
    AsyncGraphDatabase = None
    Neo4jContainer = None
    PostgresContainer = None
    asyncpg = None

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "m6-hybrid-graph-retrieval-v1.json"
IMPROVEMENT_TARGET_PCT = 15.0

DIM = 128


def _make_deterministic_embedding(text: str, dim: int = DIM) -> list[float]:
    """Create a deterministic pseudo-embedding from text using hash-based projection.

    This is NOT a real embedding model. It produces a deterministic dense vector
    so that repeated queries for the same entity name always return the same
    vector. Real evaluation would use text-embedding-3-small via the
    EmbeddingProvider port.
    """
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = random.Random(h)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


async def _get_object_name(repo: Neo4jGraphRepository, entity_id: UUID) -> str | None:
    """Fetch entity name by ID."""
    entity = await repo.get_entity(entity_id)
    return entity.canonical_name if entity else None


async def _setup_neo4j_fixtures(driver: Any) -> dict[str, UUID]:
    """Populate Neo4j with multi-hop fixture data."""
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


async def _setup_pgvector_chunks(
    pg_conn: Any,
    entity_ids: dict[str, UUID],
    entity_defs: list[tuple[str, str]],
) -> dict[str, UUID]:
    """Insert entity name chunks into pgvector so vector retrieval has something to search.

    Each entity name is stored as a chunk with a deterministic embedding.
    This lets us run true vector similarity search as the "vector-only" strategy.
    """
    chunk_ids: dict[str, UUID] = {}
    index_version_id = uuid4()

    await pg_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await pg_conn.execute(
        "CREATE TABLE IF NOT EXISTS index_versions ("
        "id UUID PRIMARY KEY,"
        "index_name TEXT NOT NULL,"
        "version_number INTEGER NOT NULL,"
        "embedding_model TEXT NOT NULL,"
        "embedding_dimension INTEGER NOT NULL,"
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
        "is_active BOOLEAN NOT NULL DEFAULT true)"
    )
    await pg_conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        "chunk_id UUID PRIMARY KEY,"
        "index_version_id UUID NOT NULL REFERENCES index_versions(id),"
        "content TEXT NOT NULL,"
        "source_id UUID NOT NULL,"
        "sequence_number INTEGER NOT NULL,"
        "metadata JSONB DEFAULT '{}')"
    )
    await pg_conn.execute(
        "CREATE TABLE IF NOT EXISTS chunk_embeddings ("
        "chunk_id UUID PRIMARY KEY REFERENCES chunks(chunk_id),"
        "embedding vector(128) NOT NULL)"
    )
    await pg_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_emb ON chunk_embeddings "
        "USING ivfflat (embedding vector_cosine_ops)"
    )

    await pg_conn.execute(
        "INSERT INTO index_versions "
        "(id, index_name, version_number, embedding_model, embedding_dimension, is_active) "
        "VALUES ($1, $2, $3, $4, $5, true) "
        "ON CONFLICT (id) DO NOTHING",
        index_version_id,
        "eval-index",
        1,
        "deterministic-hash",
        DIM,
    )

    for name, etype in entity_defs:
        eid = entity_ids[name]
        chunk_id = uuid4()
        chunk_ids[name] = chunk_id
        content = f"{name} ({etype})"
        embedding = _make_deterministic_embedding(content)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        await pg_conn.execute(
            "INSERT INTO chunks "
            "(chunk_id, index_version_id, content, source_id, sequence_number, metadata) "
            "VALUES ($1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (chunk_id) DO NOTHING",
            chunk_id,
            index_version_id,
            content,
            eid,
            0,
            json.dumps({"entity_type": etype}),
        )
        await pg_conn.execute(
            "INSERT INTO chunk_embeddings (chunk_id, embedding) "
            "VALUES ($1, $2::vector) "
            "ON CONFLICT (chunk_id) DO NOTHING",
            chunk_id,
            embedding_str,
        )

    return chunk_ids


async def _vector_search(
    pg_conn: Any,
    query_text: str,
    top_k: int,
    tenant_id: str,
) -> list[str]:
    """Run pgvector similarity search and return matched entity names.

    Uses the deterministic embedding of the query text to find similar
    entity name chunks.
    """
    query_embedding = _make_deterministic_embedding(query_text)
    query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    rows = await pg_conn.fetch(
        """
        SELECT c.content, c.metadata
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
        ORDER BY ce.embedding <=> $1::vector
        LIMIT $2
        """,
        query_str,
        top_k,
    )
    return [row["content"] for row in rows]


async def _vector_search_with_scores(
    pg_conn: Any,
    query_text: str,
    top_k: int,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Run pgvector similarity search and return matched chunk info with scores."""
    query_embedding = _make_deterministic_embedding(query_text)
    query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    rows = await pg_conn.fetch(
        """
        SELECT c.content, c.metadata, c.chunk_id
        FROM chunk_embeddings ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
        ORDER BY ce.embedding <=> $1::vector
        LIMIT $2
        """,
        query_str,
        top_k,
    )
    return [
        {"content": row["content"], "chunk_id": row["chunk_id"], "metadata": row["metadata"]}
        for row in rows
    ]


async def _evaluate_case(
    case: dict[str, Any],
    repo: Neo4jGraphRepository,
    entity_ids: dict[str, UUID],
    pg_conn: Any | None,
) -> dict[str, Any]:
    """Evaluate a single multi-hop case using three retrieval strategies:

    1. Vector-only:   pgvector similarity search using the actual user question
    2. Graph-only:    Neo4j graph traversal at max_depth=2 (no vector component)
    3. Hybrid:        True fusion of pgvector results + graph traversal via RRF

    The metric is whether the expected_object_entity name appears in the
    retrieved results (binary recall) and how many results are returned (precision).
    """
    svc = GraphFusionService(graph_repository=repo)

    seed_name = case["seed_entity"]
    seed_id = entity_ids.get(seed_name)
    if seed_id is None:
        return {
            "case_id": case["id"],
            "case_type": case["type"],
            "error": f"Seed entity {seed_name} not found",
        }

    expected_object = case.get("expected_object_entity")
    max_depth = case.get("max_depth", 2)
    valid_at_str = case.get("valid_at")
    valid_at = datetime.fromisoformat(valid_at_str) if valid_at_str else None
    principal = case.get("principal", "engineering")
    tenant_id = case.get("tenant_id", "eval-tenant")

    entity_seed = CanonicalEntity(
        entity_id=seed_id,
        entity_type="Service",
        canonical_name=seed_name,
        aliases=[],
        attributes={},
    )

    # Strategy 1: Vector-only (pgvector similarity search using actual user question)
    vector_found_names: list[str] = []
    vector_chunks: list[dict[str, Any]] = []
    if pg_conn is not None:
        query_text = case["question"]
        vector_chunks = await _vector_search_with_scores(
            pg_conn, query_text, top_k=5, tenant_id=tenant_id
        )
        vector_found_names = [
            r["content"]
            for r in vector_chunks
            if expected_object and expected_object in r["content"]
        ]
    vector_expected_found = expected_object in vector_found_names if expected_object else False
    vector_recall = 1.0 if vector_expected_found else 0.0

    # Strategy 2: Graph-only (Neo4j graph traversal at max_depth)
    graph_evidence = await svc.retrieve_evidence(
        seed_entities=[entity_seed],
        predicates=[case.get("expected_predicate")] if case.get("expected_predicate") else None,
        valid_at=valid_at,
        max_depth=max_depth,
        tenant_id=tenant_id,
        principal=principal,
    )

    found_object_names: list[str] = []
    for ev in graph_evidence:
        if ev.graph_path_fact_ids:
            last_fact_id = ev.graph_path_fact_ids[-1]
            obj_id = await _get_object_id_from_fact(repo, last_fact_id, seed_id)
            if obj_id:
                obj_name = await _get_object_name(repo, obj_id)
                if obj_name:
                    found_object_names.append(obj_name)

    expected_found = expected_object in found_object_names if expected_object else False
    graph_recall = 1.0 if expected_found else 0.0
    graph_precision = 1.0 / len(found_object_names) if found_object_names else 0.0

    # Strategy 3: True Hybrid - combining vector + graph via RRF fusion
    hybrid_found_names: list[str] = []
    if pg_conn is not None:
        vector_results = [
            _FakeRetrievedChunk(
                chunk_id=r["chunk_id"],
                source_id=r["chunk_id"],
                document_id=None,
                version_id=None,
                content=r["content"],
                vector_score=1.0,
                keyword_score=None,
                allowed_principals=[principal],
            )
            for r in vector_chunks
        ]
        fused_evidence = hybrid_rrf_fusion(vector_results, [], graph_evidence)
        hybrid_found_names = [
            ev.content for ev in fused_evidence if expected_object and expected_object in ev.content
        ]

    hybrid_expected_found = expected_object in hybrid_found_names if expected_object else False
    hybrid_recall = 1.0 if hybrid_expected_found else 0.0

    if hybrid_recall > vector_recall > 0:
        relative_improvement = (hybrid_recall - vector_recall) / vector_recall * 100
    elif hybrid_recall > vector_recall and vector_recall == 0:
        relative_improvement = None
    elif hybrid_recall == vector_recall:
        relative_improvement = 0.0
    else:
        relative_improvement = 0.0

    return {
        "case_id": case["id"],
        "case_type": case["type"],
        "vector_recall": vector_recall,
        "graph_only_recall": graph_recall,
        "hybrid_recall": hybrid_recall,
        "graph_only_precision": graph_precision,
        "relative_improvement_pct": relative_improvement,
        "expected_object": expected_object,
        "vector_found": vector_found_names,
        "graph_only_found": found_object_names,
        "hybrid_found": hybrid_found_names,
        "expected_object_found": expected_found,
        "num_graph_evidence": len(graph_evidence),
    }


async def _get_object_id_from_fact(
    repo: Neo4jGraphRepository, fact_id: UUID, default_id: UUID
) -> UUID | None:
    """Get the object entity ID from a fact by its ID."""
    fact = await repo.get_fact(fact_id)
    return fact.object_id if fact else default_id


async def run_evaluation() -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Run the M6 hybrid evaluation against Neo4j + pgvector Testcontainers."""
    if (
        AsyncGraphDatabase is None
        or Neo4jContainer is None
        or PostgresContainer is None
        or asyncpg is None
    ):
        return {
            "status": "skipped",
            "reason": "neo4j/postgres driver or testcontainers not available",
            "cases": [],
        }

    dataset = json.loads(DATASET_PATH.read_text())
    cases = dataset["cases"]

    neo4j_container: Any | None = None
    pg_container: Any | None = None
    driver: Any | None = None
    pg_conn: Any | None = None

    try:
        neo4j_container = Neo4jContainer()
        neo4j_container.start()
        neo4j_host = neo4j_container.get_container_host_ip()
        bolt_port = neo4j_container.get_exposed_port(7687)
        neo4j_uri = f"bolt://{neo4j_host}:{int(bolt_port)}"
        neo4j_user = neo4j_container.username
        neo4j_password = neo4j_container.password
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Neo4j Docker not available: {exc}",
            "cases": [],
        }

    try:
        pg_container = PostgresContainer(image="pgvector/pgvector:pg16")
        pg_container.start()
        pg_host = pg_container.get_container_host_ip()
        pg_port = int(pg_container.get_exposed_port(5432))
    except Exception as exc:
        if neo4j_container:
            neo4j_container.stop()
        return {
            "status": "skipped",
            "reason": f"pgvector Docker not available: {exc}",
            "cases": [],
        }

    try:
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        await driver.verify_connectivity()

        entity_ids = await _setup_neo4j_fixtures(driver)
        repo = Neo4jGraphRepository(driver=driver, database="neo4j")

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

        pg_conn = await asyncpg.connect(
            host=pg_host,
            port=pg_port,
            user="test",
            password="test",
            database="test",
        )
        _chunk_ids = await _setup_pgvector_chunks(pg_conn, entity_ids, entity_defs)

        results: list[dict[str, Any]] = []
        for case in cases:
            result = await _evaluate_case(case, repo, entity_ids, pg_conn)
            results.append(result)

        vector_recalls = [r["vector_recall"] for r in results if r.get("vector_recall") is not None]
        graph_only_recalls = [
            r["graph_only_recall"] for r in results if r.get("graph_only_recall") is not None
        ]
        hybrid_recalls = [r["hybrid_recall"] for r in results if r.get("hybrid_recall") is not None]
        improvements = [
            r["relative_improvement_pct"]
            for r in results
            if r.get("relative_improvement_pct") is not None
        ]

        avg_vector_recall = sum(vector_recalls) / len(vector_recalls) if vector_recalls else 0.0
        avg_graph_only_recall = (
            sum(graph_only_recalls) / len(graph_only_recalls) if graph_only_recalls else 0.0
        )
        avg_hybrid_recall = sum(hybrid_recalls) / len(hybrid_recalls) if hybrid_recalls else 0.0
        avg_improvement_pct = sum(improvements) / len(improvements) if improvements else 0.0

        multi_hop_cases = [r for r in results if "multi-hop" in r.get("case_type", "")]
        if multi_hop_cases:
            mh_vector = [
                r["vector_recall"] for r in multi_hop_cases if r.get("vector_recall") is not None
            ]
            mh_hybrid = [
                r["hybrid_recall"] for r in multi_hop_cases if r.get("hybrid_recall") is not None
            ]
            mh_avg_vector = sum(mh_vector) / len(mh_vector) if mh_vector else 0.0
            mh_avg_hybrid = sum(mh_hybrid) / len(mh_hybrid) if mh_hybrid else 0.0
            if mh_avg_vector > 0:
                mh_relative_improvement = ((mh_avg_hybrid - mh_avg_vector) / mh_avg_vector) * 100
            elif mh_avg_hybrid > 0 and mh_avg_vector == 0:
                mh_relative_improvement = None
            else:
                mh_relative_improvement = 0.0
        else:
            mh_avg_vector = 0.0
            mh_avg_hybrid = 0.0
            mh_relative_improvement = None

        return {
            "status": "completed",
            "dataset_version": dataset["version"],
            "strategies_evaluated": ["vector_only", "graph_only", "hybrid"],
            "total_cases": len(cases),
            "evaluated_cases": len([r for r in results if r.get("error") is None]),
            "avg_vector_recall": round(avg_vector_recall, 3),
            "avg_graph_only_recall": round(avg_graph_only_recall, 3),
            "avg_hybrid_recall": round(avg_hybrid_recall, 3),
            "relative_improvement_pct": round(avg_improvement_pct, 3)
            if avg_improvement_pct
            else None,
            "multi_hop_relative_improvement_pct": (
                round(mh_relative_improvement, 3) if mh_relative_improvement is not None else None
            ),
            "meets_15_percent_target": (
                mh_relative_improvement is not None
                and mh_relative_improvement >= IMPROVEMENT_TARGET_PCT
            ),
            "cases": results,
        }
    finally:
        if pg_conn is not None:
            await pg_conn.close()
        if driver is not None:
            await driver.close()
        if neo4j_container is not None:
            neo4j_container.stop()
        if pg_container is not None:
            pg_container.stop()


if __name__ == "__main__":
    result = asyncio.run(run_evaluation())
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["status"] == "completed" else 1)
