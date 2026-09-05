"""Neo4j graph repository adapter.

Implements the GraphRepository port for Neo4j using the official async driver.
"""

from __future__ import annotations

import json
from typing import Any, LiteralString, cast
from uuid import UUID

from neo4j import AsyncDriver

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.domain.types import validate_json_value

_ENTITY_MERGE: LiteralString = """
MERGE (e:Entity {entity_id: $entity_id})
SET e.entity_type = $entity_type,
    e.canonical_name = $canonical_name,
    e.aliases = $aliases,
    e.attributes = $attributes
RETURN e
"""

_ENTITY_FIND: LiteralString = """
MATCH (e:Entity)
WHERE ($canonical_name IS NULL OR e.canonical_name = $canonical_name)
  AND ($entity_type IS NULL OR e.entity_type = $entity_type)
RETURN e
LIMIT 100
"""

_FACT_MERGE: LiteralString = """
MERGE (f:Fact {fact_id: $fact_id})
SET f.subject_id = $subject_id,
    f.predicate = $predicate,
    f.object_id = $object_id,
    f.status = $status,
    f.confidence = $confidence,
    f.evidence_ids = $evidence_ids,
    f.valid_from = $valid_from,
    f.valid_to = $valid_to,
    f.observed_at = $observed_at,
    f.extraction_method = $extraction_method,
    f.ontology_version = $ontology_version
WITH f
MATCH (s:Entity {entity_id: $subject_id})
MERGE (s)-[r:SUBJECT_OF]->(f)
WITH f
MATCH (o:Entity {entity_id: $object_id})
MERGE (f)-[r2:OBJECT]->(o)
RETURN f
"""

_FACT_UPDATE_STATUS: LiteralString = """
MATCH (f:Fact {fact_id: $fact_id})
SET f.status = $status
OPTIONAL MATCH (f)-[r:SUBJECT_OF]->(s)
OPTIONAL MATCH (f)-[r2:OBJECT]->(o)
OPTIONAL MATCH (new:Fact {fact_id: $superseded_by})
FOREACH (_ IN CASE WHEN $superseded_by IS NOT NULL THEN [1] ELSE [] END |
    MERGE (f)-[:SUPERSEDED_BY]->(new)
)
RETURN f
"""

_MENTION_MERGE: LiteralString = """
MERGE (m:Mention {mention_id: $mention_id})
SET m.chunk_id = $chunk_id,
    m.surface_form = $surface_form,
    m.candidate_type = $candidate_type,
    m.locator = $locator,
    m.extraction_confidence = $extraction_confidence
WITH m
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (m)-[:MENTIONED_IN]->(c)
RETURN m
"""


def _dict_to_entity(record: Any) -> CanonicalEntity:
    raw = record.data() if hasattr(record, "data") else record
    node = raw.get("e", raw)
    node_dict: dict[str, Any] = node if isinstance(node, dict) else {}
    return CanonicalEntity(
        entity_id=cast(UUID, node_dict.get("entity_id")),
        entity_type=cast(str, node_dict.get("entity_type")),
        canonical_name=cast(str, node_dict.get("canonical_name")),
        aliases=list(node_dict.get("aliases") or []),
        attributes=json.loads(cast(str, node_dict.get("attributes")))
        if node_dict.get("attributes")
        else {},
    )


def _neo4j_datetime_to_native(value: Any) -> Any:
    if hasattr(value, "to_native"):
        return value.to_native()
    return value


def _dict_to_fact(node: dict[str, Any]) -> KnowledgeFact:
    return KnowledgeFact(
        fact_id=node["fact_id"],
        subject_id=node["subject_id"],
        predicate=node["predicate"],
        object_id=node["object_id"],
        status=node["status"],
        confidence=node["confidence"],
        evidence_ids=node.get("evidence_ids", []),
        valid_from=_neo4j_datetime_to_native(node.get("valid_from")),
        valid_to=_neo4j_datetime_to_native(node.get("valid_to")),
        observed_at=_neo4j_datetime_to_native(node["observed_at"]),
        extraction_method=node["extraction_method"],
        ontology_version=node["ontology_version"],
    )


class Neo4jGraphRepository:
    """Neo4j implementation of GraphRepository port."""

    def __init__(self, driver: AsyncDriver, tx: Any | None = None) -> None:
        self._driver = driver
        self._tx = tx

    async def _run_read(self, cypher: str, **params: Any) -> Any:
        if self._tx is not None:
            return await self._tx.run(cypher, **params)
        async with cast(Any, self._driver.session()) as session:
            return await session.run(cypher, **params)

    async def _run_write(self, cypher: str, **params: Any) -> None:
        if self._tx is not None:
            await self._tx.run(cypher, **params)
            return
        async with (
            cast(Any, self._driver.session()) as session,
            await session.begin_transaction() as tx,
        ):
            await tx.run(cypher, **params)
            await tx.commit()

    async def create_entity(self, entity: CanonicalEntity) -> CanonicalEntity:
        params: dict[str, Any] = {
            "entity_id": str(entity.entity_id),
            "entity_type": entity.entity_type,
            "canonical_name": entity.canonical_name,
            "aliases": entity.aliases,
            "attributes": json.dumps(validate_json_value(entity.attributes))
            if entity.attributes
            else "{}",
        }
        await self._run_write(_ENTITY_MERGE, **params)
        return entity

    async def get_entity(self, entity_id: UUID) -> CanonicalEntity | None:
        cypher = "MATCH (e:Entity {entity_id: $entity_id}) RETURN e"
        result = await self._run_read(cypher, entity_id=str(entity_id))
        record = await result.single()
        if not record:
            return None
        return _dict_to_entity(record)

    async def find_entities(
        self, canonical_name: str | None = None, entity_type: str | None = None
    ) -> list[CanonicalEntity]:
        params: dict[str, Any] = {"canonical_name": canonical_name, "entity_type": entity_type}
        result = await self._run_read(_ENTITY_FIND, **params)
        records = await result.data()
        return [_dict_to_entity(r) for r in records]

    async def create_fact(self, fact: KnowledgeFact) -> KnowledgeFact:
        params: dict[str, Any] = {
            "fact_id": str(fact.fact_id),
            "subject_id": str(fact.subject_id),
            "predicate": fact.predicate,
            "object_id": str(fact.object_id),
            "status": fact.status,
            "confidence": fact.confidence,
            "evidence_ids": [str(e) for e in fact.evidence_ids],
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "observed_at": fact.observed_at,
            "extraction_method": fact.extraction_method,
            "ontology_version": fact.ontology_version,
        }
        await self._run_write(_FACT_MERGE, **params)
        return fact

    async def get_fact(self, fact_id: UUID) -> KnowledgeFact | None:
        cypher = "MATCH (f:Fact {fact_id: $fact_id}) RETURN f"
        result = await self._run_read(cypher, fact_id=str(fact_id))
        record = await result.single()
        if not record:
            return None
        return _dict_to_fact(record["f"])

    async def find_facts(
        self,
        subject_id: UUID | None = None,
        predicate: str | None = None,
        object_id: UUID | None = None,
        status: str | None = None,
    ) -> list[KnowledgeFact]:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if subject_id:
            conditions.append("startNode(r).entity_id = $subject_id")
            params["subject_id"] = str(subject_id)
        if predicate:
            conditions.append("f.predicate = $predicate")
            params["predicate"] = predicate
        if object_id:
            conditions.append("endNode(r2).entity_id = $object_id")
            params["object_id"] = str(object_id)
        if status:
            conditions.append("f.status = $status")
            params["status"] = status

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        cypher = f"""
        MATCH (s:Entity)-[r:SUBJECT_OF]->(f:Fact)-[r2:OBJECT]->(o:Entity)
        {where_clause}
        RETURN f
        LIMIT 100
        """
        result = await self._run_read(cypher, **params)  # pyright: ignore[reportArgumentType]
        records = await result.data()
        return [_dict_to_fact(r["f"]) for r in records]

    async def update_fact_status(
        self,
        fact_id: UUID,
        status: str,
        superseded_by: UUID | None = None,
    ) -> KnowledgeFact:
        params: dict[str, Any] = {
            "fact_id": str(fact_id),
            "status": status,
            "superseded_by": str(superseded_by) if superseded_by else None,
        }
        await self._run_write(_FACT_UPDATE_STATUS, **params)
        fact = await self.get_fact(fact_id)
        if not fact:
            raise ValueError(f"Fact {fact_id} not found after status update")
        return fact

    async def create_mention(self, mention: EntityMention) -> EntityMention:
        params: dict[str, Any] = {
            "mention_id": str(mention.mention_id),
            "chunk_id": str(mention.chunk_id),
            "surface_form": mention.surface_form,
            "candidate_type": mention.candidate_type,
            "locator": mention.locator,
            "extraction_confidence": mention.extraction_confidence,
        }
        await self._run_write(_MENTION_MERGE, **params)
        return mention

    async def find_mentions(self, chunk_id: UUID) -> list[EntityMention]:
        cypher = """
        MATCH (m:Mention)-[:MENTIONED_IN]->(c:Chunk {chunk_id: $chunk_id})
        RETURN m
        """
        result = await self._run_read(cypher, chunk_id=str(chunk_id))
        records = await result.data()
        return [
            EntityMention(
                mention_id=r["m"]["mention_id"],
                chunk_id=r["m"]["chunk_id"],
                surface_form=r["m"]["surface_form"],
                candidate_type=r["m"]["candidate_type"],
                locator=r["m"].get("locator"),
                extraction_confidence=r["m"]["extraction_confidence"],
            )
            for r in records
        ]
