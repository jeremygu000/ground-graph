"""M3 document source_locator - stable identity for concurrent ingestion safety

Revision ID: 003_document_source_locator
Revises: 002_identity_constraints
Create Date: 2026-09-06

Adds source_locator to documents table for DB-enforced document identity
and a UNIQUE(source_id, source_locator) constraint to prevent concurrent
duplicate document creation.

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "003_document_source_locator"
down_revision: str | None = "002_identity_constraints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN source_locator TEXT NOT NULL DEFAULT ''")
    op.execute("UPDATE documents SET source_locator = ''")
    op.create_unique_constraint(
        "uq_documents_source_locator",
        "documents",
        ["source_id", "source_locator"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_documents_source_locator",
        "documents",
        type_="unique",
    )
    op.execute("ALTER TABLE documents DROP COLUMN source_locator")
