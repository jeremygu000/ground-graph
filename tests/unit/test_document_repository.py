"""Unit tests for the PostgreSQL document repository."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from groundgraph.domain.documents import Chunk, ParsedDocument, SourceDescriptor
from groundgraph.infrastructure.postgres.document_repository import PostgresDocumentRepository
from groundgraph.infrastructure.postgres.models import (
    Chunk as SqlChunk,
)
from groundgraph.infrastructure.postgres.models import (
    Document as SqlDocument,
)
from groundgraph.infrastructure.postgres.models import (
    DocumentVersion as SqlDocumentVersion,
)
from groundgraph.infrastructure.postgres.models import (
    Source as SqlSource,
)


@dataclass
class _Result:
    row: object | None = None
    rows: list[object] | None = None

    def scalar_one_or_none(self) -> object | None:
        return self.row

    def scalars(self) -> _Result:
        return self

    def all(self) -> list[object]:
        return list(self.rows or [])


class _Session:
    def __init__(self, responses: Sequence[_Result] | None = None) -> None:
        self.added: list[object] = []
        self.executed: list[object] = []
        self.flushed = 0
        self.responses = list(responses or [])

    def add(self, instance: object) -> None:
        self.added.append(instance)

    async def execute(self, statement: object, parameters: object | None = None) -> _Result:
        self.executed.append(statement)
        if self.responses:
            return self.responses.pop(0)
        return _Result()

    async def flush(self) -> None:
        self.flushed += 1


def _source_model(source_id: UUID, uri: str) -> SqlSource:
    return SqlSource(
        source_id=source_id,
        source_type="filesystem",
        uri=uri,
        classification="internal",
        tenant_id="tenant-a",
        allowed_principals=["engineering"],
    )


def _document_model(
    document_id: UUID,
    source_id: UUID,
    *,
    current_version_id: UUID | None,
) -> SqlDocument:
    return SqlDocument(
        document_id=document_id,
        source_id=source_id,
        title="Doc",
        media_type="text/markdown",
        current_version_id=current_version_id,
    )


def _version_model(
    version_id: UUID,
    document_id: UUID,
    *,
    is_current: bool,
    created_at: datetime,
) -> SqlDocumentVersion:
    return SqlDocumentVersion(
        version_id=version_id,
        document_id=document_id,
        checksum="abc123",
        content="# Hello",
        doc_metadata={"author": "test"},
        effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        is_current=is_current,
        created_at=created_at,
    )


def _chunk_model(
    chunk_id: UUID,
    document_id: UUID,
    version_id: UUID,
    *,
    ordinal: int,
    heading_path: list[str],
    allowed_principals: list[str],
) -> SqlChunk:
    return SqlChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=version_id,
        ordinal=ordinal,
        heading_path=heading_path,
        content=f"chunk-{ordinal}",
        token_count=ordinal + 1,
        checksum=f"chunk-{ordinal}",
        allowed_principals=allowed_principals,
    )


@pytest.mark.asyncio
async def test_document_repository_happy_paths() -> None:
    source_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    chunk_id = uuid4()

    source_row = _source_model(source_id, "/docs/a")
    document_row = _document_model(document_id, source_id, current_version_id=version_id)
    version_row = _version_model(
        version_id,
        document_id,
        is_current=True,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    older_version_row = _version_model(
        uuid4(),
        document_id,
        is_current=False,
        created_at=datetime(2023, 12, 1, tzinfo=UTC),
    )
    chunk_row = _chunk_model(
        chunk_id,
        document_id,
        version_id,
        ordinal=0,
        heading_path=["Intro"],
        allowed_principals=["engineering"],
    )
    later_chunk_row = _chunk_model(
        uuid4(),
        document_id,
        version_id,
        ordinal=1,
        heading_path=["More"],
        allowed_principals=["engineering"],
    )

    session = _Session(
        responses=[
            _Result(row=None),  # create_document existing lookup
            _Result(),  # create_document upsert
            _Result(row=source_row),
            _Result(rows=[source_row, _source_model(uuid4(), "/docs/b")]),
            _Result(row=document_row),
            _Result(row=version_row),
            _Result(row=document_row),
            _Result(row=version_row),
            _Result(row=document_row),
            _Result(rows=[version_row, older_version_row]),
            _Result(row=chunk_row),
            _Result(rows=[chunk_row, later_chunk_row]),
            _Result(),
            _Result(),
        ]
    )
    repo = PostgresDocumentRepository(cast(Any, session))

    created_source = SourceDescriptor(
        source_id=source_id,
        source_type="filesystem",
        uri="/docs/a",
        classification="internal",
        tenant_id="tenant-a",
        allowed_principals=["engineering"],
    )
    created_document = ParsedDocument(
        document_id=document_id,
        version_id=version_id,
        source_id=source_id,
        title="Doc",
        media_type="text/markdown",
        checksum="abc123",
        content="# Hello",
        metadata={"author": "test"},
        effective_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    created_chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=version_id,
        ordinal=0,
        heading_path=["Intro"],
        content="chunk-0",
        token_count=1,
        checksum="chunk-0",
        allowed_principals=["engineering"],
    )

    assert await repo.create_source(created_source) == created_source
    assert await repo.create_document(created_document) == created_document
    assert await repo.create_chunk(created_chunk) == created_chunk

    loaded_source = await repo.get_source(source_id)
    assert loaded_source is not None
    assert loaded_source.uri == "/docs/a"

    sources = await repo.list_sources()
    assert len(sources) == 2
    assert {source.uri for source in sources} == {"/docs/a", "/docs/b"}

    loaded_document = await repo.get_document(document_id)
    assert loaded_document is not None
    assert loaded_document.version_id == version_id

    loaded_version = await repo.get_document_version(document_id, version_id)
    assert loaded_version is not None
    assert loaded_version.checksum == "abc123"

    versions = await repo.list_document_versions(document_id)
    assert [version.version_id for version in versions] == [
        version_id,
        older_version_row.version_id,
    ]

    loaded_chunk = await repo.get_chunk(chunk_id)
    assert loaded_chunk is not None
    assert loaded_chunk.heading_path == ["Intro"]

    chunks = await repo.list_chunks(document_id, version_id)
    assert [chunk.ordinal for chunk in chunks] == [0, 1]

    await repo.delete_document(document_id)

    assert session.flushed == 3
    assert len(session.added) == 3
    assert len(session.executed) == 14


@pytest.mark.asyncio
async def test_document_repository_falls_back_to_current_version_flag() -> None:
    source_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()

    document_row = _document_model(document_id, source_id, current_version_id=None)
    version_row = _version_model(
        version_id,
        document_id,
        is_current=True,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    session = _Session(
        responses=[
            _Result(row=document_row),
            _Result(row=version_row),
        ]
    )
    repo = PostgresDocumentRepository(cast(Any, session))

    loaded = await repo.get_document(document_id)
    assert loaded is not None
    assert loaded.version_id == version_id


@pytest.mark.asyncio
async def test_document_repository_missing_rows_return_none_or_empty() -> None:
    source_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    chunk_id = uuid4()

    document_row = _document_model(document_id, source_id, current_version_id=version_id)
    session = _Session(
        responses=[
            _Result(row=None),  # get_source
            _Result(row=document_row),  # get_document
            _Result(row=None),  # get current version lookup fails
            _Result(row=None),  # get_document_version document lookup
            _Result(row=document_row),  # list_document_versions document lookup
            _Result(rows=[]),  # list_document_versions empty list
            _Result(row=None),  # get_chunk
            _Result(rows=[]),  # list_chunks
            _Result(),  # delete_document
            _Result(),  # delete_document version update
        ]
    )
    repo = PostgresDocumentRepository(cast(Any, session))

    assert await repo.get_source(source_id) is None
    assert await repo.get_document(document_id) is None
    assert await repo.get_document_version(document_id, version_id) is None
    assert await repo.list_document_versions(document_id) == []
    assert await repo.get_chunk(chunk_id) is None
    assert await repo.list_chunks(document_id, version_id) == []

    await repo.delete_document(document_id)
