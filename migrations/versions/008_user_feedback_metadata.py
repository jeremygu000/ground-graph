"""M12 add category and notes to user_feedback

Revision ID: 008_user_feedback_metadata
Revises: 007_source_deactivation
Create Date: 2026-09-07

Adds category (VARCHAR 100 NULL) and notes (TEXT NULL) columns to the
user_feedback table to support structured feedback categorization and
operator notes for the M10/M12 review workflow.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "008_user_feedback_metadata"
down_revision: str | None = "007_source_deactivation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_feedback",
        sa.Column("category", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "user_feedback",
        sa.Column("notes", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_feedback", "notes")
    op.drop_column("user_feedback", "category")
