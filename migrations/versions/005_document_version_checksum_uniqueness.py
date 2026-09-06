"""M5 document_versions (document_id, checksum) uniqueness for concurrent idempotency

Revision ID: 005_version_checksum_uniqueness
Revises: 004_ingestion_checkpoint
Create Date: 2026-09-06

Adds unique constraint on (document_id, checksum) in document_versions to
prevent concurrent workers from creating duplicate versions for the same
document + content. This enforces plan.md requirement: same source + same
checksum must not create a new version.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "005_version_checksum_uniqueness"
down_revision: str | None = "004_ingestion_checkpoint"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_document_versions_document_id_checksum",
        "document_versions",
        ["document_id", "checksum"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_document_versions_document_id_checksum",
        "document_versions",
    )
