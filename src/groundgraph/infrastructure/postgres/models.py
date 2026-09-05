"""SQLAlchemy async models for PostgreSQL (plan.md §5.1).

These models implement the PostgreSQL side of the persistence model.
They are intentionally not in the domain layer — they belong to the
infrastructure layer.

All UUID primary keys, UTC timestamps, and proper FK/DeletionBehavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

try:
    from pgvector.sqlalchemy import VECTOR
except ImportError:  # pragma: no cover
    raise ImportError("pgvector is required: pip install pgvector") from None


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map: ClassVar[dict[type, Any]] = {
        UUID: PG_UUID,
        list[str]: ARRAY(String),
        list[UUID]: ARRAY(PG_UUID),
        dict[str, Any]: JSONB,
    }


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _gen_uuid() -> UUID:
    return uuid4()


class Source(Base):
    __tablename__ = "sources"

    source_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    classification: Mapped[str] = mapped_column(String(100), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    allowed_principals: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=_utcnow
    )

    documents: Mapped[list[Document]] = relationship(
        "Document", back_populates="source", cascade="all, delete-orphan"
    )


class SourceSyncState(Base):
    __tablename__ = "source_sync_state"

    id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    source_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False, unique=True
    )
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    sync_status: Mapped[str] = mapped_column(String(50), nullable=False, default="idle")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=_utcnow
    )


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    source_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    current_version_id: Mapped[UUID | None] = mapped_column(PG_UUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=_utcnow
    )

    source: Mapped[Source] = relationship("Source", back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        "DocumentVersion",
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        passive_deletes="all",
        order_by="DocumentVersion.created_at.desc()",
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="document", passive_deletes="all"
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "current_version_id"],
            ["document_versions.document_id", "document_versions.version_id"],
            ondelete="SET NULL",
            use_alter=True,
            name="fk_documents_current_version_id_document_versions",
            deferrable=True,
            initially="DEFERRED",
        ),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    version_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("documents.document_id", ondelete="RESTRICT"), nullable=False
    )
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="versions",
        foreign_keys=[document_id],
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="version",
        cascade="all, delete-orphan",
        overlaps="chunks",
        foreign_keys="[Chunk.document_id, Chunk.version_id]",
    )

    __table_args__ = (
        Index("ix_document_versions_document_id", "document_id"),
        Index("ix_document_versions_checksum", "checksum"),
        UniqueConstraint(
            "document_id",
            "version_id",
            name="uq_document_versions_document_id_version_id",
        ),
        Index(
            "ix_document_versions_current",
            "document_id",
            unique=True,
            postgresql_where=sa.text("is_current = true"),
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("documents.document_id", ondelete="RESTRICT"), nullable=False
    )
    version_id: Mapped[UUID] = mapped_column(PG_UUID, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str] = mapped_column(Text, nullable=False)
    start_locator: Mapped[str | None] = mapped_column(Text, nullable=True)
    end_locator: Mapped[str | None] = mapped_column(Text, nullable=True)
    allowed_principals: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="chunks",
        overlaps="chunks",
    )
    version: Mapped[DocumentVersion] = relationship(
        "DocumentVersion",
        back_populates="chunks",
        overlaps="chunks,document",
        foreign_keys=[document_id, version_id],
    )
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        "ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan"
    )

    @property
    def embedding(self) -> ChunkEmbedding | None:
        return self.embeddings[0] if self.embeddings else None

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_version_id", "version_id"),
        Index("ix_chunks_checksum", "checksum"),
        ForeignKeyConstraint(
            ["document_id", "version_id"],
            ["document_versions.document_id", "document_versions.version_id"],
            ondelete="CASCADE",
        ),
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    index_version_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("index_versions.version_id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(VECTOR(1536), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunk: Mapped[Chunk] = relationship("Chunk", back_populates="embeddings")


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    source_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    documents_ingested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    steps: Mapped[list[IngestionStep]] = relationship(
        "IngestionStep", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_ingestion_runs_source_id", "source_id"),
        Index("ix_ingestion_runs_status", "status"),
    )


class IngestionStep(Base):
    __tablename__ = "ingestion_steps"

    step_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[IngestionRun] = relationship("IngestionRun", back_populates="steps")

    __table_args__ = (Index("ix_ingestion_steps_run_id", "run_id"),)


class ExecutionRun(Base):
    __tablename__ = "execution_runs"

    run_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    workflow: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    principal: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    steps: Mapped[list[ExecutionStep]] = relationship(
        "ExecutionStep", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_execution_runs_workflow", "workflow"),
        Index("ix_execution_runs_status", "status"),
        Index("ix_execution_runs_tenant_id", "tenant_id"),
    )


class ExecutionStep(Base):
    __tablename__ = "execution_steps"

    step_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("execution_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    depends_on: Mapped[list[UUID]] = mapped_column(nullable=False, default=list)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[ExecutionRun] = relationship("ExecutionRun", back_populates="steps")

    __table_args__ = (Index("ix_execution_steps_run_id", "run_id"),)


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"

    result_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("execution_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    evidence_id: Mapped[UUID] = mapped_column(PG_UUID, nullable=False)
    retrieval_method: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_retrieval_results_run_id", "run_id"),
        Index("ix_retrieval_results_evidence_id", "evidence_id"),
    )


class AnswerClaim(Base):
    __tablename__ = "answer_claims"

    claim_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("execution_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    factual: Mapped[bool] = mapped_column(Boolean, nullable=False)
    support_status: Mapped[str] = mapped_column(String(50), nullable=False)
    evidence_ids: Mapped[list[UUID]] = mapped_column(nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    citations: Mapped[list[Citation]] = relationship(
        "Citation", back_populates="claim", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_answer_claims_run_id", "run_id"),)


class Citation(Base):
    __tablename__ = "citations"

    citation_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    claim_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("answer_claims.claim_id", ondelete="CASCADE"), nullable=False
    )
    evidence_id: Mapped[UUID] = mapped_column(PG_UUID, nullable=False)
    locator: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    claim: Mapped[AnswerClaim] = relationship("AnswerClaim", back_populates="citations")

    __table_args__ = (Index("ix_citations_claim_id", "claim_id"),)


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    feedback_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("execution_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    vote: Mapped[str] = mapped_column(String(20), nullable=False)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_user_feedback_run_id", "run_id"),)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    version_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    bundle_id: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_template: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_prompt_versions_bundle_id", "bundle_id"),
        UniqueConstraint("bundle_id", "version", name="uq_prompt_versions_bundle_version"),
    )


class ModelConfigVersion(Base):
    __tablename__ = "model_config_versions"

    version_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    config_key: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_model_config_versions_config_key", "config_key"),
        UniqueConstraint("config_key", "version", name="uq_model_config_key_version"),
    )


class IndexVersion(Base):
    __tablename__ = "index_versions"

    version_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    index_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_index_versions_index_name", "index_name"),
        Index(
            "ix_index_versions_active",
            "index_name",
            unique=True,
            postgresql_where=sa.text("is_active = true"),
        ),
        UniqueConstraint("index_name", "version", name="uq_index_name_version"),
    )


class Outbox(Base):
    __tablename__ = "outbox"

    event_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    aggregate_type: Mapped[str] = mapped_column(String(50), nullable=False)
    aggregate_id: Mapped[UUID] = mapped_column(PG_UUID, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    claimed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    claim_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_outbox_status_available_created", "status", "available_at", "created_at"),
        Index("ix_outbox_aggregate", "aggregate_type", "aggregate_id"),
    )


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    dataset_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    cases: Mapped[list[EvaluationCase]] = relationship(
        "EvaluationCase", back_populates="dataset", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("name", "version", name="uq_eval_dataset_name_version"),)


class EvaluationCase(Base):
    __tablename__ = "evaluation_cases"

    case_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    dataset_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("evaluation_datasets.dataset_id", ondelete="CASCADE"), nullable=False
    )
    case_key: Mapped[str] = mapped_column(String(100), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(50), nullable=False)
    answerable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    required_evidence_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    acceptable_evidence_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    valid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    principal_scope: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    human_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    dataset: Mapped[EvaluationDataset] = relationship("EvaluationDataset", back_populates="cases")
    results: Mapped[list[EvaluationResult]] = relationship(
        "EvaluationResult", back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_evaluation_cases_dataset_id", "dataset_id"),
        UniqueConstraint("dataset_id", "case_key", name="uq_eval_case_dataset_key"),
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    run_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    dataset_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("evaluation_datasets.dataset_id", ondelete="CASCADE"), nullable=False
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    results: Mapped[list[EvaluationResult]] = relationship(
        "EvaluationResult", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_evaluation_runs_dataset_id", "dataset_id"),
        Index("ix_evaluation_runs_status", "status"),
    )


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    result_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    run_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("evaluation_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[UUID] = mapped_column(
        PG_UUID, ForeignKey("evaluation_cases.case_id", ondelete="CASCADE"), nullable=False
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    run: Mapped[EvaluationRun] = relationship("EvaluationRun", back_populates="results")
    case: Mapped[EvaluationCase] = relationship("EvaluationCase", back_populates="results")

    __table_args__ = (
        Index("ix_evaluation_results_run_id", "run_id"),
        Index("ix_evaluation_results_case_id", "case_id"),
    )


class HumanReviewItem(Base):
    __tablename__ = "human_review_items"

    review_id: Mapped[UUID] = mapped_column(PG_UUID, primary_key=True, default=_gen_uuid)
    case_id: Mapped[UUID | None] = mapped_column(PG_UUID, nullable=True)
    run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID, ForeignKey("execution_runs.run_id", ondelete="SET NULL"), nullable=True
    )
    item_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_human_review_items_run_id", "run_id"),
        Index("ix_human_review_items_decision", "decision"),
    )
