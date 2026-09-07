"""M12 add improvement_proposals table

Revision ID: 009_improvement_proposals
Revises: 008_user_feedback_metadata
Create Date: 2026-09-07

Adds improvement_proposals table for M12 controlled improvement loop.
Tracks failure clustering, proposal configs, evaluation results,
human approval, canary deployments, and promotion/rollback.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "009_improvement_proposals"
down_revision: str | None = "008_user_feedback_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "improvement_proposals",
        sa.Column("proposal_id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.String(length=100), nullable=False),
        sa.Column("failure_cluster_id", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="PROPOSED",
        ),
        sa.Column("baseline_config", JSONB(), nullable=False, server_default="{}"),
        sa.Column("proposal_config", JSONB(), nullable=False, server_default="{}"),
        sa.Column("eval_run_id", sa.UUID(), nullable=True),
        sa.Column("eval_result", JSONB(), nullable=True),
        sa.Column("approver", sa.String(length=255), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("canary_result", JSONB(), nullable=True),
        sa.Column("deployment_result", JSONB(), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rollback_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("proposal_id"),
    )
    op.create_index("ix_improvement_proposals_tenant_id", "improvement_proposals", ["tenant_id"])
    op.create_index("ix_improvement_proposals_status", "improvement_proposals", ["status"])
    op.create_index(
        "ix_improvement_proposals_failure_cluster",
        "improvement_proposals",
        ["failure_cluster_id"],
    )
    op.create_index(
        "ix_improvement_proposals_eval_run_id",
        "improvement_proposals",
        ["eval_run_id"],
    )
    op.create_foreign_key(
        "fk_improvement_proposals_eval_run_id_evaluation_runs",
        "improvement_proposals",
        "evaluation_runs",
        ["eval_run_id"],
        ["run_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_table("improvement_proposals")
