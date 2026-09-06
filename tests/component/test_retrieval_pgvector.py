"""PostgreSQL retrieval component tests (pgvector + FTS).

These tests run against a real Postgres+pgvector container started via
Testcontainers (see ``conftest.py``). They verify:

  * pgvector similarity search with ranking
  * Pre-retrieval ACL filtering
  * Cross-tenant isolation
  * Keyword full-text search
  * Only current document version is returned

Marked as ``@pytest.mark.component`` so it can be run independently of
the docker-compose stack smoke tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from groundgraph.infrastructure.postgres.keyword_retriever import PostgresKeywordRetriever
from groundgraph.infrastructure.postgres.models import Base as PostgresBase
from groundgraph.infrastructure.postgres.models import Chunk as SqlChunk
from groundgraph.infrastructure.postgres.models import ChunkEmbedding as SqlChunkEmbedding
from groundgraph.infrastructure.postgres.models import Document as SqlDocument
from groundgraph.infrastructure.postgres.models import DocumentVersion as SqlDocumentVersion
from groundgraph.infrastructure.postgres.models import IndexVersion as SqlIndexVersion
from groundgraph.infrastructure.postgres.models import Source as SqlSource
from groundgraph.infrastructure.postgres.vector_retriever import PostgresVectorRetriever

pytestmark = [pytest.mark.integration, pytest.mark.component]


@asynccontextmanager
async def _postgres_session(
    dsn: str,
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create a function-scoped session factory for PostgreSQL tests."""
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(PostgresBase.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _seed_corpus(  # noqa: PLR0917
    session: AsyncSession,
    tenant_id: str,
    seed_id: str,
    chunk_contents: list[str],
    embedding_vectors: list[list[float]],
    allowed_principals: list[str],
) -> dict[str, Any]:
    """Seed a minimal test corpus and return IDs."""
    source = SqlSource(
        source_id=uuid4(),
        source_type="markdown",
        uri=f"internal://test/{seed_id}",
        classification="internal",
        tenant_id=tenant_id,
        allowed_principals=allowed_principals,
        is_active=True,
    )
    session.add(source)

    doc = SqlDocument(
        document_id=uuid4(),
        source_id=source.source_id,
        source_locator=f"internal://test/{seed_id}/doc.md",
        title="Test Document",
        media_type="text/markdown",
    )
    session.add(doc)

    version = SqlDocumentVersion(
        version_id=uuid4(),
        document_id=doc.document_id,
        checksum=f"checksum-{seed_id}",
        content="Test content for retrieval.",
        doc_metadata={},
        effective_at=None,
        is_current=True,
    )
    session.add(version)

    doc.current_version_id = version.version_id

    chunks = []
    for i, content in enumerate(chunk_contents):
        chunk = SqlChunk(
            chunk_id=uuid4(),
            document_id=doc.document_id,
            version_id=version.version_id,
            ordinal=i + 1,
            heading_path=[f"Section {i + 1}"],
            content=content,
            token_count=20,
            checksum=f"chunk-{i}-{seed_id}",
            allowed_principals=allowed_principals,
        )
        session.add(chunk)
        chunks.append(chunk)

    index_version = SqlIndexVersion(
        version_id=uuid4(),
        index_name=f"idx-{seed_id}",
        version="v1",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        chunker_version="v1",
        is_active=True,
    )
    session.add(index_version)
    await session.flush()

    for chunk, vec in zip(chunks, embedding_vectors, strict=True):
        emb = SqlChunkEmbedding(
            chunk_id=chunk.chunk_id,
            index_version_id=index_version.version_id,
            embedding=vec,
        )
        session.add(emb)

    await session.commit()

    return {
        "source": source,
        "document": doc,
        "version": version,
        "chunks": chunks,
        "index_version": index_version,
    }


async def test_vector_search_returns_correctly_ranked_chunks(
    postgres_component: Any,
) -> None:
    """pgvector returns higher similarity for directionally closer vectors."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        chunk_contents = [
            "API gateway handles all incoming requests",
            "PostgreSQL enables similarity search over embeddings",
            "Redis cache stores frequently accessed data",
        ]
        vec_a = [1.0] * 1536
        vec_b = [0.0, 1.0] + [0.0] * 1534
        vec_c = [0.0] * 1536
        vec_c[500] = 1.0
        ids = await _seed_corpus(
            session,
            "test-tenant",
            "rank-test",
            chunk_contents,
            [vec_a, vec_b, vec_c],
            allowed_principals=["eng"],
        )
        chunk_ids = [c.chunk_id for c in ids["chunks"]]
        index_version_id = ids["index_version"].version_id

        retriever = PostgresVectorRetriever(cast(Any, session_factory))
        results = await retriever.search_with_content(
            vec_a,
            top_k=3,
            filters={
                "tenant_id": "test-tenant",
                "allowed_principals": ["eng"],
                "index_version_id": index_version_id,
            },
        )

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_ids[0]


async def test_acl_filter_excludes_unauthorized_chunks(
    postgres_component: Any,
) -> None:
    """Chunks not in the caller's principal scope must not be returned."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        ids = await _seed_corpus(
            session,
            "test-tenant",
            "acl-test",
            ["API gateway content", "Redis cache content", "PostgreSQL content"],
            [[1.0] * 1536, [0.5] * 1536, [0.2] * 1536],
            allowed_principals=["eng"],
        )
        chunk_ids = {c.chunk_id for c in ids["chunks"]}
        index_version_id = ids["index_version"].version_id

        retriever = PostgresVectorRetriever(cast(Any, session_factory))
        results = await retriever.search_with_content(
            [1.0] * 1536,
            top_k=10,
            filters={
                "tenant_id": "test-tenant",
                "allowed_principals": ["no-access-principal"],
                "index_version_id": index_version_id,
            },
        )

        result_ids = {r.chunk_id for r in results}
        assert result_ids.isdisjoint(chunk_ids)


async def test_cross_tenant_isolation(
    postgres_component: Any,
) -> None:
    """Tenant A's principal must not see Tenant B's chunks."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        ids_a = await _seed_corpus(
            session,
            "tenant-a",
            "tenant-a-test",
            ["Tenant A specific document content"],
            [[1.0] * 1536],
            allowed_principals=["eng"],
        )
        ids_b = await _seed_corpus(
            session,
            "tenant-b",
            "tenant-b-test",
            ["Tenant B specific document content"],
            [[1.0] * 1536],
            allowed_principals=["eng"],
        )
        index_version_a = ids_a["index_version"].version_id

        retriever = PostgresVectorRetriever(cast(Any, session_factory))
        results = await retriever.search_with_content(
            [1.0] * 1536,
            top_k=5,
            filters={
                "tenant_id": "tenant-a",
                "allowed_principals": ["eng"],
                "index_version_id": index_version_a,
            },
        )

        result_ids = {r.chunk_id for r in results}
        assert ids_b["chunks"][0].chunk_id not in result_ids
        assert ids_a["chunks"][0].chunk_id in result_ids


async def test_keyword_search_returns_matching_chunks(
    postgres_component: Any,
) -> None:
    """Full-text search returns chunks containing the query terms."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        ids = await _seed_corpus(
            session,
            "test-tenant",
            "kw-test",
            [
                "The API gateway handles all incoming requests and routes them.",
                "PostgreSQL with pgvector enables similarity search over embeddings.",
                "Redis cache stores frequently accessed data for faster retrieval.",
            ],
            [[1.0] * 1536, [0.5] * 1536, [0.2] * 1536],
            allowed_principals=["eng"],
        )
        chunk_ids = [c.chunk_id for c in ids["chunks"]]

        retriever = PostgresKeywordRetriever(cast(Any, session_factory))
        results = await retriever.search(
            "PostgreSQL pgvector similarity",
            top_k=3,
            tenant_id="test-tenant",
            allowed_principals=["eng"],
        )

        assert len(results) >= 1
        assert results[0].chunk_id == chunk_ids[1]


async def test_keyword_search_respects_tenant_filter(
    postgres_component: Any,
) -> None:
    """Keyword search must not return chunks from other tenants."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        await _seed_corpus(
            session,
            "tenant-x",
            "kw-tenant-x",
            ["Tenant X private document content"],
            [[1.0] * 1536],
            allowed_principals=["eng"],
        )

        retriever = PostgresKeywordRetriever(cast(Any, session_factory))
        results = await retriever.search(
            "private document",
            top_k=5,
            tenant_id="tenant-y",
            allowed_principals=["eng"],
        )

        assert len(results) == 0


async def test_old_document_version_is_not_returned(
    postgres_component: Any,
) -> None:
    """Only chunks from the current document version are returned."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        source = SqlSource(
            source_id=uuid4(),
            source_type="markdown",
            uri="internal://multi-version-test",
            classification="internal",
            tenant_id="test-tenant",
            allowed_principals=["eng"],
            is_active=True,
        )
        session.add(source)

        doc = SqlDocument(
            document_id=uuid4(),
            source_id=source.source_id,
            source_locator="internal://multi-version-test/doc.md",
            title="Multi-Version Doc",
            media_type="text/markdown",
        )
        session.add(doc)

        old_version = SqlDocumentVersion(
            version_id=uuid4(),
            document_id=doc.document_id,
            checksum="old-checksum",
            content="Old version content about timeouts.",
            doc_metadata={},
            effective_at=None,
            is_current=False,
        )
        session.add(old_version)

        new_version = SqlDocumentVersion(
            version_id=uuid4(),
            document_id=doc.document_id,
            checksum="new-checksum",
            content="New version content about timeouts being 60 seconds.",
            doc_metadata={},
            effective_at=None,
            is_current=True,
        )
        session.add(new_version)

        doc.current_version_id = new_version.version_id

        old_chunk = SqlChunk(
            chunk_id=uuid4(),
            document_id=doc.document_id,
            version_id=old_version.version_id,
            ordinal=1,
            heading_path=["Section"],
            content="The timeout is 30 seconds.",
            token_count=10,
            checksum="old-chunk",
            allowed_principals=["eng"],
        )
        session.add(old_chunk)

        new_chunk = SqlChunk(
            chunk_id=uuid4(),
            document_id=doc.document_id,
            version_id=new_version.version_id,
            ordinal=1,
            heading_path=["Section"],
            content="The timeout is 60 seconds.",
            token_count=10,
            checksum="new-chunk",
            allowed_principals=["eng"],
        )
        session.add(new_chunk)

        index_version = SqlIndexVersion(
            version_id=uuid4(),
            index_name="idx-multi-version",
            version="v1",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            chunker_version="v1",
            is_active=True,
        )
        session.add(index_version)
        await session.flush()

        session.add(
            SqlChunkEmbedding(
                chunk_id=old_chunk.chunk_id,
                index_version_id=index_version.version_id,
                embedding=[1.0] * 1536,
            )
        )
        session.add(
            SqlChunkEmbedding(
                chunk_id=new_chunk.chunk_id,
                index_version_id=index_version.version_id,
                embedding=[1.0] * 1536,
            )
        )
        await session.commit()

        retriever = PostgresVectorRetriever(cast(Any, session_factory))
        results = await retriever.search_with_content(
            [1.0] * 1536,
            top_k=5,
            filters={
                "tenant_id": "test-tenant",
                "allowed_principals": ["eng"],
                "index_version_id": index_version.version_id,
            },
        )

        result_ids = {r.chunk_id for r in results}
        assert old_chunk.chunk_id not in result_ids
        assert new_chunk.chunk_id in result_ids
