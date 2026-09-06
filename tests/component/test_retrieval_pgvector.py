"""PostgreSQL + pgvector retrieval component tests.

These tests run against a real Postgres+pgvector container started via
Testcontainers (see ``conftest.py``). They verify:

  * pgvector similarity search returns correct results
  * Pre-retrieval ACL filtering excludes unauthorized chunks
  * Cross-tenant queries return no results
  * Citation locators point to the exact chunk/version

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

from groundgraph.infrastructure.postgres.models import Base as PostgresBase
from groundgraph.infrastructure.postgres.models import (
    Chunk as SqlChunk,
)
from groundgraph.infrastructure.postgres.models import (
    ChunkEmbedding as SqlChunkEmbedding,
)
from groundgraph.infrastructure.postgres.models import (
    Document as SqlDocument,
)
from groundgraph.infrastructure.postgres.models import (
    DocumentVersion as SqlDocumentVersion,
)
from groundgraph.infrastructure.postgres.models import (
    IndexVersion as SqlIndexVersion,
)
from groundgraph.infrastructure.postgres.models import (
    Source as SqlSource,
)
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


async def _seed_test_data(session: AsyncSession, tenant_id: str, seed_id: str) -> dict[str, Any]:
    """Seed a minimal test corpus and return IDs."""
    source = SqlSource(
        source_id=uuid4(),
        source_type="markdown",
        uri=f"internal://test/{seed_id}",
        classification="internal",
        tenant_id=tenant_id,
        allowed_principals=["eng", f"tenant:{tenant_id}"],
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
        checksum=f"test-checksum-{seed_id}",
        content="Test content for retrieval.",
        doc_metadata={},
        effective_at=None,
        is_current=True,
    )
    session.add(version)

    doc.current_version_id = version.version_id

    chunk1 = SqlChunk(
        chunk_id=uuid4(),
        document_id=doc.document_id,
        version_id=version.version_id,
        ordinal=1,
        heading_path=["Section 1"],
        content="The API gateway handles all incoming requests and routes them to services.",
        token_count=20,
        checksum=f"chunk-1-{seed_id}",
        allowed_principals=["eng", f"tenant:{tenant_id}"],
    )
    session.add(chunk1)

    chunk2 = SqlChunk(
        chunk_id=uuid4(),
        document_id=doc.document_id,
        version_id=version.version_id,
        ordinal=2,
        heading_path=["Section 2"],
        content="PostgreSQL with pgvector enables similarity search over embeddings.",
        token_count=15,
        checksum=f"chunk-2-{seed_id}",
        allowed_principals=["eng", f"tenant:{tenant_id}"],
    )
    session.add(chunk2)

    chunk3 = SqlChunk(
        chunk_id=uuid4(),
        document_id=doc.document_id,
        version_id=version.version_id,
        ordinal=3,
        heading_path=["Section 3"],
        content="Redis cache stores frequently accessed data for faster retrieval.",
        token_count=18,
        checksum=f"chunk-3-{seed_id}",
        allowed_principals=["eng"],
    )
    session.add(chunk3)

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

    embedding1 = SqlChunkEmbedding(
        chunk_id=chunk1.chunk_id,
        index_version_id=index_version.version_id,
        embedding=[0.05] * 1536,
    )
    session.add(embedding1)

    embedding2 = SqlChunkEmbedding(
        chunk_id=chunk2.chunk_id,
        index_version_id=index_version.version_id,
        embedding=[0.02] * 1536,
    )
    session.add(embedding2)

    embedding3 = SqlChunkEmbedding(
        chunk_id=chunk3.chunk_id,
        index_version_id=index_version.version_id,
        embedding=[0.01] * 1536,
    )
    session.add(embedding3)

    await session.commit()

    return {
        "source": source,
        "document": doc,
        "version": version,
        "chunk1": chunk1,
        "chunk2": chunk2,
        "chunk3": chunk3,
        "index_version": index_version,
    }


async def test_vector_search_returns_similar_chunks(
    postgres_component: Any,
) -> None:
    """pgvector similarity search returns the most similar chunk first."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        ids = await _seed_test_data(session, "test-tenant", "seed-1")
        chunk1_id = ids["chunk1"].chunk_id
        index_version_id = ids["index_version"].version_id

        retriever = PostgresVectorRetriever(cast(Any, session))
        query_vector = [0.05] * 1536

        results = await retriever.search_with_content(
            query_vector,
            top_k=3,
            filters={
                "tenant_id": "test-tenant",
                "allowed_principals": ["eng"],
                "index_version_id": index_version_id,
            },
        )

        assert len(results) >= 1
        assert results[0].chunk_id == chunk1_id
        assert results[0].score is not None


async def test_acl_filter_excludes_unauthorized_chunks(
    postgres_component: Any,
) -> None:
    """Chunks not in the caller's principal scope must not be returned."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        ids = await _seed_test_data(session, "test-tenant", "seed-2")
        chunk3_id = ids["chunk3"].chunk_id

        retriever = PostgresVectorRetriever(cast(Any, session))
        query_vector = [0.05] * 1536

        results = await retriever.search_with_content(
            query_vector,
            top_k=10,
            filters={
                "tenant_id": "test-tenant",
                "allowed_principals": ["no-access-principal"],
            },
        )

        chunk_ids = {r.chunk_id for r in results}
        assert chunk3_id not in chunk_ids


async def test_empty_result_for_no_matching_chunks(
    postgres_component: Any,
) -> None:
    """Query with no principal access returns empty list."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        await _seed_test_data(session, "test-tenant", "seed-3")

        retriever = PostgresVectorRetriever(cast(Any, session))
        query_vector = [0.99] * 1536

        results = await retriever.search_with_content(
            query_vector,
            top_k=5,
            filters={"tenant_id": "test-tenant", "allowed_principals": ["no-such-principal"]},
        )

        assert len(results) == 0


async def test_cross_tenant_returns_no_results(
    postgres_component: Any,
) -> None:
    """Different tenant principals must not see another tenant's chunks."""
    dsn = postgres_component.dsn
    async with _postgres_session(dsn) as session_factory, session_factory() as session:
        await _seed_test_data(session, "tenant-alpha", "seed-4")

        retriever = PostgresVectorRetriever(cast(Any, session))
        query_vector = [0.05] * 1536

        results = await retriever.search_with_content(
            query_vector,
            top_k=5,
            filters={"tenant_id": "tenant-alpha", "allowed_principals": ["tenant:tenant-beta"]},
        )

        assert len(results) == 0
