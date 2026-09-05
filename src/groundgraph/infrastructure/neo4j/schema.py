"""Neo4j schema management utility (plan.md §5.2).

Creates constraints and indexes on startup. Safe to re-run —
existing constraints are not duplicated.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase

from groundgraph.application.settings import get_settings

logger = structlog.get_logger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    (
        "CREATE CONSTRAINT version_id IF NOT EXISTS FOR (v:DocumentVersion)"
        " REQUIRE v.version_id IS UNIQUE"
    ),
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    (
        "CREATE CONSTRAINT ontology_type_id IF NOT EXISTS FOR (t:OntologyType)"
        " REQUIRE t.type_id IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT ontology_predicate_id IF NOT EXISTS FOR (p:OntologyPredicate)"
        " REQUIRE p.predicate_id IS UNIQUE"
    ),
]

INDEXES = [
    "CREATE INDEX entity_canonical_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_aliases IF NOT EXISTS FOR (e:Entity) ON (e.aliases)",
    "CREATE INDEX fact_predicate IF NOT EXISTS FOR (f:Fact) ON (f.predicate)",
    "CREATE INDEX fact_status IF NOT EXISTS FOR (f:Fact) ON (f.status)",
    "CREATE INDEX fact_validity IF NOT EXISTS FOR (f:Fact) ON (f.valid_from, f.valid_to)",
    "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
    "CREATE INDEX chunk_version IF NOT EXISTS FOR (c:Chunk) ON (c.version_id)",
    "CREATE INDEX mention_chunk IF NOT EXISTS FOR (m:Mention) ON (m.chunk_id)",
    "CREATE INDEX mention_surface_form IF NOT EXISTS FOR (m:Mention) ON (m.surface_form)",
]


async def ensure_schema(driver: AsyncDriver) -> None:
    """Create all constraints and indexes if they don't exist.

    Idempotent — safe to call on every startup.
    """
    settings = get_settings()
    logger.info("neo4j.schema.ensure", database=settings.neo4j_database)

    async with cast(Any, driver.session(database=settings.neo4j_database)) as session:
        for constraint_cypher in CONSTRAINTS:
            try:
                await session.run(constraint_cypher)  # type: ignore[arg-type]
                logger.debug("neo4j.schema.constraint_created", cypher=constraint_cypher)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    logger.warning(
                        "neo4j.schema.constraint_failed", cypher=constraint_cypher, exc=exc
                    )

        for index_cypher in INDEXES:
            try:
                await session.run(index_cypher)  # type: ignore[arg-type]
                logger.debug("neo4j.schema.index_created", cypher=index_cypher)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    logger.warning("neo4j.schema.index_failed", cypher=index_cypher, exc=exc)

    logger.info("neo4j.schema.ready")


async def create_driver() -> AsyncDriver:
    """Create and return an async Neo4j driver."""
    settings = get_settings()
    return AsyncGraphDatabase.driver(  # pyright: ignore[reportUnknownMemberType]
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
    )
