"""M3 identity constraints - source unique by tenant+type+uri

Revision ID: 002_identity_constraints
Revises: 001_initial
Create Date: 2026-09-06

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "002_identity_constraints"
down_revision: str | None = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_sources_tenant_type_uri",
        "sources",
        ["tenant_id", "source_type", "uri"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_sources_tenant_type_uri",
        "sources",
        type_="unique",
    )
