"""initial schema - sources, documents, chunks, executions, outbox

Revision ID: 001_initial
Revises:
Create Date: 2026-09-05

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # sources
    op.create_table(
        "sources",
        sa.Column("source_id", postgresql.UUID(), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("classification", sa.String(length=100), nullable=False),
        sa.Column(
            "allowed_principals",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("source_id"),
    )

    # source_sync_state
    op.create_table(
        "source_sync_state",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("source_id", postgresql.UUID(), nullable=False),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_checksum", sa.Text(), nullable=True),
        sa.Column("sync_status", sa.String(length=50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id"),
    )

    # documents
    op.create_table(
        "documents",
        sa.Column("document_id", postgresql.UUID(), nullable=False),
        sa.Column("source_id", postgresql.UUID(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False),
        sa.Column("current_version_id", postgresql.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
    )

    # document_versions
    op.create_table(
        "document_versions",
        sa.Column("version_id", postgresql.UUID(), nullable=False),
        sa.Column("document_id", postgresql.UUID(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("doc_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_checksum", "document_versions", ["checksum"])

    # chunks
    op.create_table(
        "chunks",
        sa.Column("chunk_id", postgresql.UUID(), nullable=False),
        sa.Column("document_id", postgresql.UUID(), nullable=False),
        sa.Column("version_id", postgresql.UUID(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "heading_path",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("start_locator", sa.Text(), nullable=True),
        sa.Column("end_locator", sa.Text(), nullable=True),
        sa.Column(
            "allowed_principals",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id"], ["document_versions.version_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_version_id", "chunks", ["version_id"])
    op.create_index("ix_chunks_checksum", "chunks", ["checksum"])

    # chunk_embeddings (pgvector)
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", postgresql.UUID(), nullable=False),
        sa.Column("embedding", postgresql.VECTOR(dim=1536), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("index_version", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.chunk_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunk_embeddings_index_version", "chunk_embeddings", ["index_version"])

    # ingestion_runs
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("source_id", postgresql.UUID(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("documents_ingested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errors", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_ingestion_runs_source_id", "ingestion_runs", ["source_id"])
    op.create_index("ix_ingestion_runs_status", "ingestion_runs", ["status"])

    # ingestion_steps
    op.create_table(
        "ingestion_steps",
        sa.Column("step_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("input", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("output", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["ingestion_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("step_id"),
    )
    op.create_index("ix_ingestion_steps_run_id", "ingestion_steps", ["run_id"])

    # execution_runs
    op.create_table(
        "execution_runs",
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("workflow", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("principal", sa.String(length=255), nullable=False),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("input", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("output", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_execution_runs_workflow", "execution_runs", ["workflow"])
    op.create_index("ix_execution_runs_status", "execution_runs", ["status"])
    op.create_index("ix_execution_runs_tenant_id", "execution_runs", ["tenant_id"])

    # execution_steps
    op.create_table(
        "execution_steps",
        sa.Column("step_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "depends_on",
            postgresql.ARRAY(postgresql.UUID()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("input", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("output", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("step_id"),
    )
    op.create_index("ix_execution_steps_run_id", "execution_steps", ["run_id"])

    # retrieval_results
    op.create_table(
        "retrieval_results",
        sa.Column("result_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(), nullable=False),
        sa.Column("retrieval_method", sa.String(length=50), nullable=False),
        sa.Column("vector_score", sa.Float(), nullable=True),
        sa.Column("rerank_score", sa.Float(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("result_id"),
    )
    op.create_index("ix_retrieval_results_run_id", "retrieval_results", ["run_id"])
    op.create_index("ix_retrieval_results_evidence_id", "retrieval_results", ["evidence_id"])

    # answer_claims
    op.create_table(
        "answer_claims",
        sa.Column("claim_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("factual", sa.Boolean(), nullable=False),
        sa.Column("support_status", sa.String(length=50), nullable=False),
        sa.Column(
            "evidence_ids",
            postgresql.ARRAY(postgresql.UUID()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("claim_id"),
    )
    op.create_index("ix_answer_claims_run_id", "answer_claims", ["run_id"])

    # citations
    op.create_table(
        "citations",
        sa.Column("citation_id", postgresql.UUID(), nullable=False),
        sa.Column("claim_id", postgresql.UUID(), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(), nullable=False),
        sa.Column("locator", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["claim_id"], ["answer_claims.claim_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("citation_id"),
    )
    op.create_index("ix_citations_claim_id", "citations", ["claim_id"])

    # user_feedback
    op.create_table(
        "user_feedback",
        sa.Column("feedback_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("vote", sa.String(length=20), nullable=False),
        sa.Column("correction", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("feedback_id"),
    )
    op.create_index("ix_user_feedback_run_id", "user_feedback", ["run_id"])

    # outbox
    op.create_table(
        "outbox",
        sa.Column("event_id", postgresql.UUID(), nullable=False),
        sa.Column("aggregate_type", sa.String(length=50), nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_outbox_status_created", "outbox", ["status", "created_at"])
    op.create_index("ix_outbox_aggregate", "outbox", ["aggregate_type", "aggregate_id"])

    # prompt_versions
    op.create_table(
        "prompt_versions",
        sa.Column("version_id", postgresql.UUID(), nullable=False),
        sa.Column("bundle_id", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("prompt_template", postgresql.JSONB(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index("ix_prompt_versions_bundle_id", "prompt_versions", ["bundle_id"])
    op.create_unique_constraint(
        "uq_prompt_versions_bundle_version", "prompt_versions", ["bundle_id", "version"]
    )

    # model_config_versions
    op.create_table(
        "model_config_versions",
        sa.Column("version_id", postgresql.UUID(), nullable=False),
        sa.Column("config_key", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index("ix_model_config_versions_config_key", "model_config_versions", ["config_key"])
    op.create_unique_constraint(
        "uq_model_config_key_version", "model_config_versions", ["config_key", "version"]
    )

    # index_versions
    op.create_table(
        "index_versions",
        sa.Column("version_id", postgresql.UUID(), nullable=False),
        sa.Column("index_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("chunker_version", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("version_id"),
    )
    op.create_index("ix_index_versions_index_name", "index_versions", ["index_name"])
    op.create_unique_constraint(
        "uq_index_name_version", "index_versions", ["index_name", "version"]
    )

    # evaluation_datasets
    op.create_table(
        "evaluation_datasets",
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("dataset_id"),
    )
    op.create_unique_constraint(
        "uq_eval_dataset_name_version", "evaluation_datasets", ["name", "version"]
    )

    # evaluation_cases
    op.create_table(
        "evaluation_cases",
        sa.Column("case_id", postgresql.UUID(), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("case_key", sa.String(length=100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=50), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column(
            "required_evidence_ids",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "acceptable_evidence_ids",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("valid_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "principal_scope",
            postgresql.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("tags", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("human_notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["evaluation_datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("case_id"),
    )
    op.create_index("ix_evaluation_cases_dataset_id", "evaluation_cases", ["dataset_id"])
    op.create_unique_constraint(
        "uq_eval_case_dataset_key", "evaluation_cases", ["dataset_id", "case_key"]
    )

    # evaluation_runs
    op.create_table(
        "evaluation_runs",
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["evaluation_datasets.dataset_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_evaluation_runs_dataset_id", "evaluation_runs", ["dataset_id"])
    op.create_index("ix_evaluation_runs_status", "evaluation_runs", ["status"])

    # evaluation_results
    op.create_table(
        "evaluation_results",
        sa.Column("result_id", postgresql.UUID(), nullable=False),
        sa.Column("run_id", postgresql.UUID(), nullable=False),
        sa.Column("case_id", postgresql.UUID(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["evaluation_runs.run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["case_id"], ["evaluation_cases.case_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("result_id"),
    )
    op.create_index("ix_evaluation_results_run_id", "evaluation_results", ["run_id"])
    op.create_index("ix_evaluation_results_case_id", "evaluation_results", ["case_id"])

    # human_review_items
    op.create_table(
        "human_review_items",
        sa.Column("review_id", postgresql.UUID(), nullable=False),
        sa.Column("case_id", postgresql.UUID(), nullable=True),
        sa.Column("run_id", postgresql.UUID(), nullable=True),
        sa.Column("item_type", sa.String(length=50), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("decision", sa.String(length=20), nullable=True),
        sa.Column("decided_by", sa.String(length=255), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.run_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("review_id"),
    )
    op.create_index("ix_human_review_items_run_id", "human_review_items", ["run_id"])
    op.create_index("ix_human_review_items_decision", "human_review_items", ["decision"])


def downgrade() -> None:
    op.drop_table("human_review_items")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_cases")
    op.drop_table("evaluation_datasets")
    op.drop_table("index_versions")
    op.drop_table("model_config_versions")
    op.drop_table("prompt_versions")
    op.drop_table("outbox")
    op.drop_table("user_feedback")
    op.drop_table("citations")
    op.drop_table("answer_claims")
    op.drop_table("retrieval_results")
    op.drop_table("execution_steps")
    op.drop_table("execution_runs")
    op.drop_table("ingestion_steps")
    op.drop_table("ingestion_runs")
    op.drop_table("chunk_embeddings")
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("source_sync_state")
    op.drop_table("sources")
