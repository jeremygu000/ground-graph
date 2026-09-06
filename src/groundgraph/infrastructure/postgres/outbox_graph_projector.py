"""Outbox consumer that projects entity/fact events to Neo4j."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from groundgraph.application.ports import OutboxRepository
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType
from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.neo4j.unit_of_work import Neo4jUnitOfWork

logger = logging.getLogger(__name__)

WORKER_ID = "graph-projector"
LEASE_DURATION_SECONDS = 30


class OutboxGraphProjector:
    """Consumes outbox events and projects them to Neo4j.

    Handles:
      - ENTITY_MENTIONED → create mention in Neo4j
      - ENTITY_RESOLVED → create/update canonical entity in Neo4j
      - FACT_CANDIDATE → create fact in Neo4j
    """

    def __init__(self, graph_repository: Neo4jGraphRepository) -> None:
        self._graph = graph_repository

    async def project_event(self, event: OutboxEvent) -> None:
        """Project a single outbox event to Neo4j."""
        event_type = event.event_type
        payload = event.payload

        try:
            if event_type == OutboxEventType.ENTITY_MENTIONED:
                await self._project_mention(payload)
            elif event_type == OutboxEventType.ENTITY_RESOLVED:
                await self._project_entity(payload)
            elif event_type == OutboxEventType.FACT_CANDIDATE:
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
        valid_from = payload.get("valid_from")
        valid_to = payload.get("valid_to")
        if isinstance(valid_from, str):
            valid_from = datetime.fromisoformat(valid_from)
        if isinstance(valid_to, str):
            valid_to = datetime.fromisoformat(valid_to)
        fact = KnowledgeFact(
            fact_id=payload["fact_id"],
            subject_id=payload["subject_id"],
            predicate=payload["predicate"],
            object_id=payload["object_id"],
            status=payload.get("status", "candidate"),
            confidence=payload.get("confidence", 0.5),
            evidence_ids=payload.get("evidence_ids", []),
            valid_from=valid_from,
            valid_to=valid_to,
            observed_at=payload.get("observed_at") or datetime.now(UTC),
            extraction_method=payload.get("extraction_method", "llm"),
            ontology_version=payload.get("ontology_version", "v0.1.0"),
            tenant_id=payload["tenant_id"],
            allowed_principals=payload.get("allowed_principals", []),
        )
        await self._graph.create_fact(fact)


class OutboxGraphWorker:
    """Background worker that polls the Postgres outbox via OutboxRepository and projects to Neo4j.

    Uses the full OutboxRepository contract with worker_id + lease + claim_token
    for stale-worker protection (see OutboxRepository port, plan.md §2.2).
    """

    def __init__(
        self,
        projector: OutboxGraphProjector,
        poll_interval: float = 1.0,
    ) -> None:
        self._projector = projector
        self._poll_interval = poll_interval
        self._running = False

    async def start(
        self,
        outbox_repo: OutboxRepository,
        uow_factory: Callable[[], Neo4jUnitOfWork],
    ) -> None:
        """Start the worker loop using claim_batch / mark_completed / mark_failed."""
        self._running = True
        while self._running:
            try:
                events = await outbox_repo.claim_batch(
                    batch_size=100,
                    worker_id=WORKER_ID,
                    lease_duration_seconds=LEASE_DURATION_SECONDS,
                )
                if events:
                    for event in events:
                        await self._process_event(outbox_repo, event, uow_factory)
                await asyncio.sleep(self._poll_interval)
            except Exception:
                logger.exception("Worker error")
                await asyncio.sleep(5)

    async def _process_event(
        self,
        outbox_repo: OutboxRepository,
        event: OutboxEvent,
        uow_factory: Callable[[], Neo4jUnitOfWork],
    ) -> None:
        uow = uow_factory()
        try:
            async with uow:
                assert uow.graph is not None, "Neo4jUnitOfWork.graph must not be None"
                projector = OutboxGraphProjector(uow.graph)
                await projector.project_event(event)
            await outbox_repo.mark_completed(event.event_id, event.claim_token)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to project event %s: %s", event.event_id, exc)
            await outbox_repo.mark_failed(event.event_id, event.claim_token, str(exc))  # type: ignore[arg-type]

    def stop(self) -> None:
        """Stop the worker loop."""
        self._running = False
