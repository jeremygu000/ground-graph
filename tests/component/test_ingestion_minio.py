"""Component tests for full ingestion pipeline: real Postgres + real MinIO.

This exercises the complete document ingestion flow with real infrastructure,
verifying:
- raw bytes land in MinIO
- document/version/chunks exist in Postgres
- exactly one outbox event is emitted
- same-checksum re-ingest produces no new version
- changed bytes produce new version but same document
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from testcontainers.community.minio import MinioContainer

from groundgraph.application.ingestion.services import IngestionService
from groundgraph.application.ports import IngestionUnitOfWork
from groundgraph.application.settings import Settings
from groundgraph.domain.documents import SourceDescriptor
from groundgraph.infrastructure.object_storage.s3_store import S3ObjectStore
from groundgraph.infrastructure.postgres.models import Base
from groundgraph.infrastructure.postgres.unit_of_work import PostgresUnitOfWork

pytestmark = [pytest.mark.integration, pytest.mark.component]


@asynccontextmanager
async def _pg_session(
    dsn: str,
) -> AsyncGenerator[async_sessionmaker[Any], None]:
    engine = create_async_engine(dsn)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


def _write_file(path: str, content: str) -> None:
    with open(path, "w") as fh:
        fh.write(content)


@pytest.fixture
def minio_endpoint(docker_available: bool) -> Any:
    if not docker_available:
        pytest.skip("Docker is not available")

    container = MinioContainer(access_key="minioadmin", secret_key="miniopassword")
    container.start()
    try:
        mc = container.get_client()
        bucket = "groundgraph-raw"
        if not mc.bucket_exists(bucket):
            mc.make_bucket(bucket)

        config = container.get_config()
        endpoint = config["endpoint"]
        yield f"http://{endpoint}"
    finally:
        container.stop()


@pytest.fixture
def s3_store(minio_endpoint: Any) -> S3ObjectStore:
    settings = Settings(
        s3_endpoint_url=minio_endpoint,
        s3_access_key="minioadmin",
        s3_secret_key=SecretStr("miniopassword"),
        s3_bucket_raw="groundgraph-raw",
    )
    return S3ObjectStore(settings)


@pytest.mark.asyncio
async def test_full_pipeline_ingest_to_s3_and_pg(
    postgres_component: Any,
    s3_store: S3ObjectStore,
) -> None:
    async with _pg_session(postgres_component.dsn) as sf:

        def make_uow() -> IngestionUnitOfWork:
            return PostgresUnitOfWork(sf)  # type: ignore[return-value]

        service = IngestionService(uow_factory=make_uow, object_store=s3_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_id = uuid4()

            async with make_uow() as uow:
                source = SourceDescriptor(
                    source_id=source_id,
                    source_type="filesystem",
                    uri=tmpdir,
                    classification="internal",
                    tenant_id="tenant-pg-minio",
                    allowed_principals=["engineering"],
                )
                await uow.documents.find_or_create_source(source)
                await uow.commit()

            test_file = os.path.join(tmpdir, "doc.md")
            await asyncio.to_thread(_write_file, test_file, "# Hello\n\nWorld content.")

            result = await service.ingest_file(
                source_id=source_id,
                file_path=test_file,
                media_type="text/markdown",
            )

            assert result.document_id is not None
            assert result.version_id is not None
            assert result.created_new_version is True

            raw_key = f"sources/{source_id}/{result.document_id}/{result.version_id}/raw"
            raw_exists = await s3_store.exists(raw_key)
            assert raw_exists, "raw bytes should exist in MinIO"

            async with make_uow() as uow:
                doc = await uow.documents.get_document(result.document_id)
                assert doc is not None
                chunks = await uow.documents.list_chunks(result.document_id, result.version_id)
                assert len(chunks) > 0
                pending = await uow.outbox.claim_batch(
                    batch_size=10,
                    worker_id="test-worker",
                    lease_duration_seconds=60,
                )
                doc_events = [e for e in pending if e.aggregate_id == result.document_id]
                assert len(doc_events) == 1


@pytest.mark.asyncio
async def test_idempotent_reingest_same_checksum(
    postgres_component: Any,
    s3_store: S3ObjectStore,
) -> None:
    async with _pg_session(postgres_component.dsn) as sf:

        def make_uow() -> IngestionUnitOfWork:
            return PostgresUnitOfWork(sf)  # type: ignore[return-value]

        service = IngestionService(uow_factory=make_uow, object_store=s3_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_id = uuid4()

            async with make_uow() as uow:
                source = SourceDescriptor(
                    source_id=source_id,
                    source_type="filesystem",
                    uri=tmpdir,
                    classification="internal",
                    tenant_id="tenant-idempotent",
                    allowed_principals=["engineering"],
                )
                await uow.documents.find_or_create_source(source)
                await uow.commit()

            test_file = os.path.join(tmpdir, "idempotent.txt")
            await asyncio.to_thread(_write_file, test_file, "Idempotent content.")

            result1 = await service.ingest_file(
                source_id=source_id,
                file_path=test_file,
                media_type="text/plain",
            )
            assert result1.created_new_version is True

            result2 = await service.ingest_file(
                source_id=source_id,
                file_path=test_file,
                media_type="text/plain",
            )
            assert result2.document_id == result1.document_id
            assert result2.version_id == result1.version_id
            assert result2.created_new_version is False


@pytest.mark.asyncio
async def test_changed_content_new_version(
    postgres_component: Any,
    s3_store: S3ObjectStore,
) -> None:
    async with _pg_session(postgres_component.dsn) as sf:

        def make_uow() -> IngestionUnitOfWork:
            return PostgresUnitOfWork(sf)  # type: ignore[return-value]

        service = IngestionService(uow_factory=make_uow, object_store=s3_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_id = uuid4()

            async with make_uow() as uow:
                source = SourceDescriptor(
                    source_id=source_id,
                    source_type="filesystem",
                    uri=tmpdir,
                    classification="internal",
                    tenant_id="tenant-versioned",
                    allowed_principals=["engineering"],
                )
                await uow.documents.find_or_create_source(source)
                await uow.commit()

            test_file = os.path.join(tmpdir, "versioned.txt")

            await asyncio.to_thread(_write_file, test_file, "Version 1.")

            result1 = await service.ingest_file(
                source_id=source_id,
                file_path=test_file,
                media_type="text/plain",
            )

            await asyncio.to_thread(_write_file, test_file, "Version 2 — different!")

            result2 = await service.ingest_file(
                source_id=source_id,
                file_path=test_file,
                media_type="text/plain",
            )

            assert result2.document_id == result1.document_id
            assert result2.version_id != result1.version_id
            assert result2.created_new_version is True

            async with make_uow() as uow:
                versions = await uow.documents.list_document_versions(result1.document_id)
                assert len(versions) == 2
