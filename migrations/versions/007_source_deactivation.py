"""M7 add source deactivation fields (is_active, deactivated_at)

Revision ID: 007_source_deactivation
Revises: 006_chunker_lineage
Create Date: 2026-09-06

Adds is_active (BOOLEAN DEFAULT TRUE NOT NULL) and deactivated_at
(TIMESTAMP WITH TIME ZONE NULL) to the sources table to support
source deactivation per plan.md §6.3 and §12.2.
Deactivated sources preserve audit history but are excluded from
active retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "007_source_deactivation"
down_revision: str | None = "006_chunker_lineage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sources",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.add_column(
        "sources",
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sources", "deactivated_at")
    op.drop_column("sources", "is_active")
