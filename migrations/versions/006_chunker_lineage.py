"""M6 add chunker_version and configuration_hash to chunks

Revision ID: 006_chunker_lineage
Revises: 005_document_version_checksum_uniqueness
Create Date: 2026-09-06

Adds chunker_version (VARCHAR(20), default 'v1') and
configuration_hash (VARCHAR(32), default '') to the chunks table
to support reproducible chunking configuration tracking per plan.md §6.4.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "006_chunker_lineage"
down_revision: str | None = "005_version_checksum_uniqueness"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("chunker_version", sa.String(length=20), nullable=False, server_default="v1"),
    )
    op.add_column(
        "chunks",
        sa.Column("configuration_hash", sa.String(length=32), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("chunks", "configuration_hash")
    op.drop_column("chunks", "chunker_version")
