"""Outbox consumer that projects entity/fact events to Neo4j."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository

logger = logging.getLogger(__name__)


class OutboxGraphProjector:
    """Consumes outbox events and projects them to Neo4j.

    Handles:
      - ENTITY_MENTIONED → create mention in Neo4j
      - ENTITY_RESOLVED → create/update canonical entity in Neo4j
      - FACT_CANDIDATE → create fact in Neo4j
    """

    def __init__(self, graph_repository: Neo4jGraphRepository) -> None:
        self._graph = graph_repository

    async def project_event(self, event: dict[str, Any]) -> None:
        """Project a single outbox event to Neo4j."""
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        try:
            if event_type == "entity_mentioned":
                await self._project_mention(payload)
            elif event_type == "entity_resolved":
                await self._project_entity(payload)
            elif event_type == "fact_candidate":
                await self._project_fact(payload)
            else:
                logger.warning("Unknown event type: %s", event_type)
        except Exception:
            logger.exception("Failed to project event %s", event_type)
            raise

    async def _project_mention(self, payload: dict[str, Any]) -> None:
        mention = EntityMention(
            mention_id=payload["mention_id"],
            chunk_id=payload["chunk_id"],
            surface_form=payload["surface_form"],
            candidate_type=payload["candidate_type"],
            locator=payload.get("locator"),
            extraction_confidence=payload.get("extraction_confidence", 0.5),
        )
        await self._graph.create_mention(mention)

    async def _project_entity(self, payload: dict[str, Any]) -> None:
        entity = CanonicalEntity(
            entity_id=payload["entity_id"],
            entity_type=payload["entity_type"],
            canonical_name=payload["canonical_name"],
            aliases=payload.get("aliases", []),
            attributes=payload.get("attributes", {}),
        )
        await self._graph.create_entity(entity)

    async def _project_fact(self, payload: dict[str, Any]) -> None:
        fact = KnowledgeFact(
            fact_id=payload["fact_id"],
            subject_id=payload["subject_id"],
            predicate=payload["predicate"],
            object_id=payload["object_id"],
            status=payload.get("status", "candidate"),
            confidence=payload.get("confidence", 0.5),
            evidence_ids=payload.get("evidence_ids", []),
            valid_from=payload.get("valid_from"),
            valid_to=payload.get("valid_to"),
            observed_at=payload.get("observed_at") or datetime.now(UTC),
            extraction_method=payload.get("extraction_method", "llm"),
            ontology_version=payload.get("ontology_version", "v0.1.0"),
        )
        await self._graph.create_fact(fact)


class OutboxGraphWorker:
    """Background worker that polls the Postgres outbox and projects to Neo4j."""

    def __init__(
        self,
        projector: OutboxGraphProjector,
        poll_interval: float = 1.0,
    ) -> None:
        self._projector = projector
        self._poll_interval = poll_interval
        self._running = False

    async def start(self, outbox_reader: Any) -> None:
        """Start the worker loop."""
        self._running = True
        while self._running:
            try:
                events = await outbox_reader.read_pending(max_count=100)
                for event in events:
                    await self._projector.project_event(event)
                await asyncio.sleep(self._poll_interval)
            except Exception:
                logger.exception("Worker error")
                await asyncio.sleep(5)

    def stop(self) -> None:
        """Stop the worker loop."""
        self._running = False
