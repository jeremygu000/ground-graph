"""Ingestion service: orchestrates file acquisition, parsing, chunking, and storage."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Tracer

from groundgraph.application.ports import IngestionUnitOfWork, ObjectStore
from groundgraph.domain.documents import (
    IngestionCheckpointStatus,
    ParsedDocument,
    SourceDescriptor,
)
from groundgraph.domain.evidence import OutboxEvent, OutboxEventType

from .chunker import Chunker
from .parsers import ParsedContent, ParserRegistry


@dataclass(frozen=True)
class IngestionReport:
    """Quality report for a single ingestion run (plan.md §6.6)."""

    source_id: UUID
    document_id: UUID
    version_id: UUID
    media_type: str
    parse_success: bool
    parse_error: str | None = None
    source_size_bytes: int = 0
    extracted_size_bytes: int = 0
    chunk_count: int = 0
    empty_chunk_count: int = 0
    duplicate_chunk_count: int = 0
    missing_acl_count: int = 0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def extracted_vs_source_ratio(self) -> float:
        if self.source_size_bytes == 0:
            return 0.0
        return round(self.extracted_size_bytes / self.source_size_bytes, 3)

    @property
    def empty_chunk_rate(self) -> float:
        if self.chunk_count == 0:
            return 0.0
        return round(self.empty_chunk_count / self.chunk_count, 3)


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    version_id: UUID
    created_new_version: bool
    tenant_id: str
    quality_report: IngestionReport | None = None


class IngestionService:
    def __init__(
        self,
        uow_factory: Callable[[], IngestionUnitOfWork],
        object_store: ObjectStore,
        chunker: Chunker | None = None,
        tracer: Tracer | None = None,
        meter: Meter | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._object_store = object_store
        self._chunker = chunker or Chunker()
        self._tracer = tracer
        self._ingestion_runs: Counter | None = None
        self._ingestion_duration: Histogram | None = None
        if meter is not None:
            self._ingestion_runs = meter.create_counter(
                "groundgraph.ingestion.runs",
                description="Ingestion runs by outcome.",
            )
            self._ingestion_duration = meter.create_histogram(
                "groundgraph.ingestion.duration",
                description="Ingestion run duration in milliseconds.",
                unit="ms",
            )

    async def ingest_file(
        self,
        source_id: UUID,
        file_path: str,
        media_type: str,
    ) -> IngestionResult:
        started = time.perf_counter()
        succeeded = False
        tracer = self._tracer
        if tracer:
            span = tracer.start_span("rag.ingestion")
            span.set_attribute("source_id", str(source_id))
            span.set_attribute("media_type", media_type)
            span.set_attribute("file_path", file_path)
        else:
            span = None
        try:
            result, source_size_bytes = await self._ingest_file_impl(
                source_id, file_path, media_type, span
            )
            duration_ms = (time.perf_counter() - started) * 1000
            report = await self.generate_report(
                source_id=source_id,
                result=result,
                source_size_bytes=source_size_bytes,
                duration_ms=duration_ms,
                media_type=media_type,
            )
            succeeded = True
            return IngestionResult(
                document_id=result.document_id,
                version_id=result.version_id,
                created_new_version=result.created_new_version,
                tenant_id=result.tenant_id,
                quality_report=report,
            )
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            if self._ingestion_runs is not None:
                self._ingestion_runs.add(1, {"status": "success" if succeeded else "failure"})
            if self._ingestion_duration is not None:
                self._ingestion_duration.record(duration_ms)
            if span:
                span.end()

    async def _ingest_file_impl(  # noqa: PLR0915
        self,
        source_id: UUID,
        file_path: str,
        media_type: str,
        span: Any,
    ) -> tuple[IngestionResult, int]:
        raw_bytes, checksum, canonical_locator, source = await self._acquire_and_validate(
            source_id, file_path, span
        )

        raw_key = f"sources/{source_id}/{checksum}/raw"
        if not await self._object_store.exists(raw_key):
            await self._object_store.put_raw(raw_key, raw_bytes, media_type)

        async with self._uow_factory() as uow:
            checkpoint = await uow.ingestion_checkpoint.get_checkpoint(
                source_id, canonical_locator, checksum
            )
            if checkpoint is not None and checkpoint.status == IngestionCheckpointStatus.PERSISTED:
                assert checkpoint.document_id is not None
                assert checkpoint.version_id is not None
                existing_doc = await uow.documents.get_document(checkpoint.document_id)
                if existing_doc is not None and existing_doc.checksum == checksum:
                    result = IngestionResult(
                        document_id=checkpoint.document_id,
                        version_id=existing_doc.version_id,
                        created_new_version=False,
                        tenant_id=source.tenant_id,
                    )
                    if span:
                        span.set_attribute("ingestion.resumed", True)
                        span.set_attribute("document_id", str(result.document_id))
                        span.set_attribute("version_id", str(result.version_id))
                    return result, len(raw_bytes)

            existing = await uow.documents.find_active_document_by_canonical_locator(
                source_id, canonical_locator
            )
            if existing is not None:
                doc_id, ver_id = existing
                current = await uow.documents.get_document_version(doc_id, ver_id)
                if current is not None and current.checksum == checksum:
                    result = IngestionResult(
                        document_id=doc_id,
                        version_id=ver_id,
                        created_new_version=False,
                        tenant_id=source.tenant_id,
                    )
                    await uow.ingestion_checkpoint.upsert_checkpoint(
                        source_id=source_id,
                        canonical_locator=canonical_locator,
                        content_checksum=checksum,
                        status=IngestionCheckpointStatus.PERSISTED,
                        document_id=doc_id,
                        version_id=ver_id,
                    )
                    if span:
                        span.set_attribute("ingestion.idempotent_hit", True)
                        span.set_attribute("document_id", str(doc_id))
                        span.set_attribute("version_id", str(ver_id))
                    return result, len(raw_bytes)
                document_id, version_id = doc_id, uuid4()
            else:
                document_id, version_id = uuid4(), uuid4()

            parsed = self._parse(raw_bytes, media_type)

            document = ParsedDocument(
                document_id=document_id,
                version_id=version_id,
                source_id=source_id,
                source_locator=canonical_locator,
                title=parsed.title,
                media_type=media_type,
                checksum=checksum,
                content=parsed.body,
                metadata={
                    "file_path": file_path,
                    "canonical_locator": canonical_locator,
                    **parsed.metadata,
                },
                effective_at=datetime.now(UTC),
            )
            canonical_doc, is_new_version = await uow.documents.upsert_document(document)
            canonical_doc_id = canonical_doc.document_id
            canonical_version_id = canonical_doc.version_id

            await uow.ingestion_checkpoint.upsert_checkpoint(
                source_id=source_id,
                canonical_locator=canonical_locator,
                content_checksum=checksum,
                status=IngestionCheckpointStatus.PERSISTED,
                document_id=canonical_doc_id,
                version_id=canonical_version_id,
            )

            if is_new_version:
                chunk_start = time.perf_counter()
                chunks = self._chunker.chunk(
                    content=parsed,
                    document_id=canonical_doc_id,
                    version_id=canonical_version_id,
                    allowed_principals=source.allowed_principals,
                )
                chunk_duration_ms = (time.perf_counter() - chunk_start) * 1000

                for chunk in chunks:
                    await uow.documents.create_chunk(chunk)

                event = OutboxEvent(
                    event_id=uuid4(),
                    aggregate_type="document",
                    aggregate_id=canonical_doc_id,
                    event_type=OutboxEventType.DOCUMENT_PARSED,
                    payload={
                        "document_id": str(canonical_doc_id),
                        "version_id": str(canonical_version_id),
                        "source_id": str(source_id),
                        "tenant_id": source.tenant_id,
                    },
                    created_at=datetime.now(UTC),
                )
                await uow.outbox.add(event)

                if span:
                    span.set_attribute("ingestion.chunk_count", len(chunks))
                    span.set_attribute("ingestion.chunk_duration_ms", round(chunk_duration_ms, 2))
                    span.set_attribute("ingestion.new_version", True)
                    span.set_attribute("document_id", str(canonical_doc_id))
                    span.set_attribute("version_id", str(canonical_version_id))
            elif span:
                span.set_attribute("ingestion.new_version", False)
                span.set_attribute("document_id", str(canonical_doc_id))
                span.set_attribute("version_id", str(canonical_version_id))

            return (
                IngestionResult(
                    document_id=canonical_doc.document_id,
                    version_id=canonical_doc.version_id,
                    created_new_version=is_new_version,
                    tenant_id=source.tenant_id,
                ),
                len(raw_bytes),
            )

    MAX_FILE_SIZE = 100 * 1024 * 1024

    async def _acquire_and_validate(
        self, source_id: UUID, file_path: str, span: Any | None = None
    ) -> tuple[bytes, str, str, SourceDescriptor]:
        async with self._uow_factory() as uow:
            source = await uow.documents.get_source(source_id)
        if source is None:
            raise ValueError(f"source not found: {source_id}")

        resolved_path = self._resolve_safe_path(source.uri, file_path)
        self._check_size(resolved_path)
        raw_bytes = await self._read_file(resolved_path)
        checksum = self._sha256(raw_bytes)
        canonical_locator = str(resolved_path)
        if span:
            span.set_attribute("content.size_bytes", len(raw_bytes))
            span.set_attribute("content.checksum", checksum[:16])
        return raw_bytes, checksum, canonical_locator, source

    def _resolve_safe_path(self, source_root: str, file_path: str) -> Path:
        root = Path(source_root).resolve()
        requested = Path(file_path)
        if requested.is_symlink():
            raise ValueError(f"symlink not allowed: {file_path}")
        requested_resolved = requested.resolve()
        try:
            requested_resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"file path outside source root: {file_path}") from exc
        return requested_resolved

    def _check_size(self, path: Path) -> None:
        size = path.stat().st_size
        if size > self.MAX_FILE_SIZE:
            raise ValueError(f"file exceeds maximum size: {size} > {self.MAX_FILE_SIZE}")

    async def _read_file(self, path: Path) -> bytes:
        def _read() -> bytes:
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    def _parse(self, content: bytes, media_type: str) -> ParsedContent:
        parser = ParserRegistry.get_with_reason(media_type)
        return parser.parse(content)

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def generate_report(  # noqa: PLR0917
        self,
        source_id: UUID,
        result: IngestionResult,
        source_size_bytes: int,
        duration_ms: float,
        parse_error: str | None = None,
        media_type: str = "",
    ) -> IngestionReport:
        """Generate an ingestion quality report (plan.md §6.6).

        This is called after ingest_file() to gather quality metrics
        from the chunks that were created.
        """
        async with self._uow_factory() as uow:
            chunks = await uow.documents.list_chunks(result.document_id, result.version_id)

        chunk_checksums: set[str] = set()
        empty_count = 0
        duplicate_count = 0
        missing_acl = 0

        for chunk in chunks:
            if not chunk.content.strip():
                empty_count += 1
            checksum = chunk.checksum
            if checksum in chunk_checksums:
                duplicate_count += 1
            else:
                chunk_checksums.add(checksum)
            if not chunk.allowed_principals:
                missing_acl += 1

        total_extracted = sum(len(c.content) for c in chunks)

        return IngestionReport(
            source_id=source_id,
            document_id=result.document_id,
            version_id=result.version_id,
            media_type=media_type,
            parse_success=parse_error is None,
            parse_error=parse_error,
            source_size_bytes=source_size_bytes,
            extracted_size_bytes=total_extracted,
            chunk_count=len(chunks),
            empty_chunk_count=empty_count,
            duplicate_chunk_count=duplicate_count,
            missing_acl_count=missing_acl,
            duration_ms=duration_ms,
        )
