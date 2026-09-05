"""M2 SQLAlchemy model import and definition tests.

Tests that SQLAlchemy models can be imported and have correct table/column definitions.
Full CRUD tests require Docker/Testcontainers (see tests/component/).
"""

from __future__ import annotations

from groundgraph.infrastructure.postgres.models import (
    Chunk,
    Document,
    DocumentVersion,
    ExecutionRun,
    IndexVersion,
    ModelConfigVersion,
    Outbox,
    PromptVersion,
    Source,
)


class TestModelImports:
    def test_all_models_importable(self) -> None:
        assert Source.__tablename__ == "sources"
        assert Document.__tablename__ == "documents"
        assert DocumentVersion.__tablename__ == "document_versions"
        assert Chunk.__tablename__ == "chunks"
        assert ExecutionRun.__tablename__ == "execution_runs"
        assert Outbox.__tablename__ == "outbox"
        assert PromptVersion.__tablename__ == "prompt_versions"
        assert ModelConfigVersion.__tablename__ == "model_config_versions"
        assert IndexVersion.__tablename__ == "index_versions"

    def test_source_model_has_required_columns(self) -> None:
        columns = {c.name for c in Source.__table__.columns}
        assert "source_id" in columns
        assert "source_type" in columns
        assert "uri" in columns
        assert "classification" in columns
        assert "allowed_principals" in columns
        assert "created_at" in columns

    def test_document_model_has_required_columns(self) -> None:
        columns = {c.name for c in Document.__table__.columns}
        assert "document_id" in columns
        assert "source_id" in columns
        assert "title" in columns
        assert "media_type" in columns
        assert "current_version_id" in columns

        source_fk = next(
            fk for fk in Document.__table__.foreign_keys if fk.parent.name == "source_id"
        )
        assert source_fk.column.table.name == "sources"
        assert source_fk.ondelete == "CASCADE"

        current_version_fk = next(
            constraint
            for constraint in Document.__table__.foreign_key_constraints
            if constraint.name == "fk_documents_current_version_id_document_versions"
        )
        assert [column.name for column in current_version_fk.columns] == [
            "document_id",
            "current_version_id",
        ]
        assert [element.column.table.name for element in current_version_fk.elements] == [
            "document_versions",
            "document_versions",
        ]
        assert current_version_fk.ondelete == "SET NULL"
        assert current_version_fk.deferrable is True
        assert current_version_fk.initially == "DEFERRED"

    def test_document_version_model_has_doc_metadata(self) -> None:
        columns = {c.name for c in DocumentVersion.__table__.columns}
        assert "version_id" in columns
        assert "document_id" in columns
        assert "checksum" in columns
        assert "content" in columns
        assert "doc_metadata" in columns
        assert "effective_at" in columns
        assert "is_current" in columns

        unique_constraint = next(
            constraint
            for constraint in DocumentVersion.__table__.constraints
            if getattr(constraint, "name", None) == "uq_document_versions_document_id_version_id"
        )
        assert [column.name for column in unique_constraint.columns] == [
            "document_id",
            "version_id",
        ]

        current_indexes = [
            index
            for index in DocumentVersion.__table__.indexes
            if index.name == "ix_document_versions_current"
        ]
        assert len(current_indexes) == 1
        assert current_indexes[0].unique is True

    def test_chunk_model_has_required_columns(self) -> None:
        columns = {c.name for c in Chunk.__table__.columns}
        assert "chunk_id" in columns
        assert "document_id" in columns
        assert "version_id" in columns
        assert "ordinal" in columns
        assert "heading_path" in columns
        assert "content" in columns
        assert "token_count" in columns
        assert "checksum" in columns
        assert "allowed_principals" in columns

        document_fk = next(
            fk
            for fk in Chunk.__table__.foreign_keys
            if fk.parent.name == "document_id" and fk.column.table.name == "documents"
        )
        version_document_fk = next(
            fk
            for fk in Chunk.__table__.foreign_keys
            if fk.parent.name == "document_id" and fk.column.table.name == "document_versions"
        )
        version_fk = next(
            constraint
            for constraint in Chunk.__table__.foreign_key_constraints
            if constraint.name is None and len(constraint.columns) == 2
        )
        assert document_fk.ondelete == "RESTRICT"
        assert version_document_fk.ondelete == "CASCADE"
        assert [column.name for column in version_fk.columns] == ["document_id", "version_id"]
        assert [element.column.table.name for element in version_fk.elements] == [
            "document_versions",
            "document_versions",
        ]
        assert version_fk.ondelete == "CASCADE"

    def test_execution_run_model_has_required_columns(self) -> None:
        columns = {c.name for c in ExecutionRun.__table__.columns}
        assert "run_id" in columns
        assert "workflow" in columns
        assert "status" in columns
        assert "principal" in columns
        assert "tenant_id" in columns
        assert "input" in columns
        assert "output" in columns
        assert "started_at" in columns
        assert "finished_at" in columns
        tenant_indexes = ExecutionRun.__table__.indexes
        assert any(index.name == "ix_execution_runs_tenant_id" for index in tenant_indexes)

    def test_outbox_model_has_required_columns(self) -> None:
        columns = {c.name for c in Outbox.__table__.columns}
        assert "event_id" in columns
        assert "aggregate_type" in columns
        assert "aggregate_id" in columns
        assert "event_type" in columns
        assert "payload" in columns
        assert "status" in columns
        assert "attempts" in columns
        assert "last_error" in columns
        assert "available_at" in columns
        assert "created_at" in columns
        assert "claimed_by" in columns
        assert "claim_token" in columns
        assert "claimed_at" in columns
        assert "lease_expires_at" in columns
        assert "completed_at" in columns

    def test_prompt_version_unique_constraint(self) -> None:
        assert any(
            "bundle_id" in str(c) and "version" in str(c)
            for c in PromptVersion.__table__.constraints
        )

    def test_model_config_version_unique_constraint(self) -> None:
        assert any(
            "config_key" in str(c) and "version" in str(c)
            for c in ModelConfigVersion.__table__.constraints
        )

    def test_index_version_unique_constraint(self) -> None:
        assert any(
            "index_name" in str(c) and "version" in str(c)
            for c in IndexVersion.__table__.constraints
        )
