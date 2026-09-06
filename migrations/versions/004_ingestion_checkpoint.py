"""M4 durable ingestion checkpoint for resumable ingestion

Revision ID: 004_ingestion_checkpoint
Revises: 003_document_source_locator
Create Date: 2026-09-06

Adds ingestion_checkpoints table to track durable ingestion state,
enabling resume without duplicating completed work after mid-ingestion
failure (plan.md §6.3: "Failed ingestion can resume from the last
durable completed step").
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_ingestion_checkpoint"
down_revision: str | None = "003_document_source_locator"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_checkpoints",
        sa.Column(
            "checkpoint_id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("content_checksum", sa.Text(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=True),
        sa.Column("version_id", sa.UUID(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("checkpoint_id"),
        sa.UniqueConstraint(
            "source_id", "content_checksum", name="uq_ingestion_checkpoints_source_checksum"
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_ingestion_checkpoints_source_id", "ingestion_checkpoints", ["source_id"])
    op.create_index("ix_ingestion_checkpoints_status", "ingestion_checkpoints", ["status"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_checkpoints_status", table_name="ingestion_checkpoints")
    op.drop_index("ix_ingestion_checkpoints_source_id", table_name="ingestion_checkpoints")
    op.drop_table("ingestion_checkpoints")
