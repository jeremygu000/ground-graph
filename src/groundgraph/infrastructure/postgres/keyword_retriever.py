"""PostgreSQL full-text search with ACL filtering."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from opentelemetry.trace import get_tracer
from sqlalchemy import func, select

from groundgraph.infrastructure.postgres.models import Chunk as ChunkModel
from groundgraph.infrastructure.postgres.models import Document as DocumentModel
from groundgraph.infrastructure.postgres.models import Source as SourceModel
from groundgraph.infrastructure.postgres.session import PostgresSession

_TRACER = get_tracer(__name__)


@dataclass
class KeywordSearchResult:
    chunk_id: UUID
    source_id: UUID
    document_id: UUID
    version_id: UUID
    content: str
    rank: float
    allowed_principals: list[str]


class PostgresKeywordRetriever:
    """PostgreSQL ``tsvector`` full-text search, ACL-filtered before results are returned."""

    def __init__(self, session: PostgresSession) -> None:
        self._session = session

    async def search(
        self,
        query: str,
        top_k: int,
        *,
        allowed_principals: list[str] | None = None,
        source_ids: list[UUID] | None = None,
        tenant_id: str,
    ) -> list[KeywordSearchResult]:
        """Return top-k keyword matches, tenant + ACL filtered.

        Uses ``websearch_to_tsquery`` so raw user input (punctuation, operators)
        does not raise ``tsquery`` syntax errors that would abort the whole
        hybrid query.
        """
        if not query or not query.strip():
            return []

        tsq = func.websearch_to_tsquery("english", query)

        conditions = [
            func.to_tsvector("english", ChunkModel.content).op("@@")(tsq),
            SourceModel.is_active == True,  # noqa: E712
            SourceModel.tenant_id == tenant_id,
            ChunkModel.version_id == DocumentModel.current_version_id,
        ]

        if allowed_principals is not None:
            conditions.append(SourceModel.allowed_principals.overlap(allowed_principals))

        if source_ids is not None:
            conditions.append(DocumentModel.source_id.in_(source_ids))

        stmt = (
            select(
                ChunkModel.chunk_id,
                DocumentModel.source_id,
                ChunkModel.document_id,
                ChunkModel.version_id,
                ChunkModel.content,
                (
                    func.ts_rank(
                        func.to_tsvector("english", ChunkModel.content),
                        tsq,
                    )
                ).label("rank"),
                SourceModel.allowed_principals,
            )
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.document_id)
            .join(SourceModel, DocumentModel.source_id == SourceModel.source_id)
            .where(*conditions)
            .order_by(func.ts_rank(func.to_tsvector("english", ChunkModel.content), tsq).desc())
            .limit(top_k)
        )

        with _TRACER.start_as_current_span("keyword.search") as span:
            span.set_attribute("keyword.top_k", top_k)
            span.set_attribute("retrieval.tenant_id", tenant_id)
            result = await self._session.execute(stmt)
            rows = result.all()
            span.set_attribute("retrieval.result_count", len(rows))
            return [
                KeywordSearchResult(
                    chunk_id=row.chunk_id,
                    source_id=row.source_id,
                    document_id=row.document_id,
                    version_id=row.version_id,
                    content=row.content,
                    rank=float(row.rank),
                    allowed_principals=list(row.allowed_principals)
                    if row.allowed_principals
                    else [],
                )
                for row in rows
            ]
