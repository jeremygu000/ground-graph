"""Repository and provider port definitions (plan.md §2.2).

These protocols define the boundaries between the domain/application layers
and the infrastructure layer. Infrastructure adapters (PostgreSQL, Neo4j,
OpenAI, S3) implement these ports.

All ports are synchronous or asynchronous interfaces using only stdlib and
Pydantic types — no framework imports allowed in the domain/application layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.domain.evidence import OutboxEvent
from groundgraph.domain.execution import ExecutionRun, ExecutionStep
from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact
from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan
from groundgraph.domain.types import JsonValue

if TYPE_CHECKING:
    from uuid import UUID


T = TypeVar("T")


class DocumentRepository(Protocol):
    """Port for document persistence operations."""

    async def create_source(self, source: SourceDescriptor) -> SourceDescriptor: ...

    async def get_source(self, source_id: UUID) -> SourceDescriptor | None: ...

    async def list_sources(self) -> list[SourceDescriptor]: ...

    async def create_document(self, document: ParsedDocument) -> ParsedDocument: ...

    async def get_document(self, document_id: UUID) -> ParsedDocument | None: ...

    async def get_document_version(
        self, document_id: UUID, version_id: UUID
    ) -> ParsedDocument | None: ...

    async def list_document_versions(self, document_id: UUID) -> list[ParsedDocument]: ...

    async def create_chunk(self, chunk: Chunk) -> Chunk: ...

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None: ...

    async def list_chunks(self, document_id: UUID, version_id: UUID) -> list[Chunk]: ...


class ObjectStore(Protocol):
    """Port for raw object storage operations (S3/MinIO)."""

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    async def get_raw(self, key: str) -> bytes: ...

    async def delete_raw(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...


class EmbeddingProvider(Protocol):
    """Port for embedding generation."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]: ...


class VectorRetriever(Protocol):
    """Port for vector similarity search."""

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[UUID, float]]: ...


class GraphRepository(Protocol):
    """Port for knowledge graph operations (Neo4j)."""

    async def create_entity(self, entity: CanonicalEntity) -> CanonicalEntity: ...

    async def get_entity(self, entity_id: UUID) -> CanonicalEntity | None: ...

    async def find_entities(
        self, canonical_name: str | None = None, entity_type: str | None = None
    ) -> list[CanonicalEntity]: ...

    async def create_fact(self, fact: KnowledgeFact) -> KnowledgeFact: ...

    async def get_fact(self, fact_id: UUID) -> KnowledgeFact | None: ...

    async def find_facts(
        self,
        subject_id: UUID | None = None,
        predicate: str | None = None,
        object_id: UUID | None = None,
        status: str | None = None,
    ) -> list[KnowledgeFact]: ...

    async def update_fact_status(
        self,
        fact_id: UUID,
        status: str,
        superseded_by: UUID | None = None,
    ) -> KnowledgeFact: ...

    async def create_mention(self, mention: EntityMention) -> EntityMention: ...

    async def find_mentions(self, chunk_id: UUID) -> list[EntityMention]: ...


class EntityExtractor(Protocol):
    """Port for extracting entities from text."""

    async def extract(self, text: str, chunk_id: UUID) -> list[EntityMention]: ...


class EntityResolver(Protocol):
    """Port for resolving entity mentions to canonical entities."""

    async def resolve(self, mention: EntityMention) -> CanonicalEntity | None: ...


class RetrievalPlanner(Protocol):
    """Port for planning retrieval strategy."""

    async def plan(self, question: str, principal: str, tenant_id: str) -> RetrievalPlan: ...


class EvidenceReranker(Protocol):
    """Port for reranking evidence."""

    async def rerank(self, evidence: list[Evidence], query: str) -> list[Evidence]: ...


class AnswerGenerator(Protocol):
    """Port for generating answers from evidence."""

    async def generate(
        self,
        question: str,
        evidence: list[Evidence],
        retrieval_plan: RetrievalPlan,
    ) -> QueryResponse: ...


class ClaimValidator(Protocol):
    """Port for validating answer claims."""

    async def validate(
        self, response: QueryResponse, evidence: list[Evidence]
    ) -> QueryResponse: ...


class TelemetryRecorder(Protocol):
    """Port for recording telemetry data."""

    def record_run(self, run: ExecutionRun) -> None: ...

    def record_step(self, step: ExecutionStep) -> None: ...

    async def flush(self) -> None: ...


class EvaluationRepository(Protocol):
    """Port for evaluation result persistence."""

    async def create_run(self, run_id: UUID, dataset_id: UUID, config: JsonValue) -> None: ...

    async def store_result(self, run_id: UUID, case_id: str, result: JsonValue) -> None: ...

    async def get_run(self, run_id: UUID) -> dict[str, Any] | None: ...


class OutboxConsumer(Protocol):
    """Port for consuming outbox events (projection worker)."""

    async def claim_batch(self, batch_size: int) -> list[OutboxEvent]: ...

    async def mark_completed(self, event_id: UUID) -> None: ...

    async def mark_failed(self, event_id: UUID, error: str) -> None: ...
