"""FastAPI dependencies and request-scoped helpers."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import suppress
from typing import Annotated, Any, cast

import asyncpg  # pyright: ignore[reportMissingTypeStubs]
import httpx
from fastapi import Depends, Header, Request
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel

from groundgraph.application.extraction.entity_resolver import EntityResolutionService
from groundgraph.application.extraction.llm_extractor import LLMEntityExtractor
from groundgraph.application.health import (
    DependencyHealth,
    HealthReasonCode,
    HealthService,
)
from groundgraph.application.retrieval.retrieval_planner import (
    RetrievalPlanner,
    RetrievalPlannerConfig,
)
from groundgraph.application.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)
from groundgraph.application.settings import Settings, get_settings
from groundgraph.infrastructure.composition import (
    PgKeywordRetrieverAdapter,
    PgVectorContentRetriever,
)
from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository
from groundgraph.infrastructure.openai.answer_generator import EvidenceOnlyAnswerGenerator
from groundgraph.infrastructure.openai.embedding_provider import OpenAIEmbeddingProvider
from groundgraph.infrastructure.openai.reranker import CrossEncoderReranker
from groundgraph.infrastructure.postgres.execution_store import ExecutionRepository
from groundgraph.infrastructure.postgres.index_version_resolver import (
    PostgresIndexVersionResolver,
)
from groundgraph.infrastructure.postgres.keyword_retriever import PostgresKeywordRetriever
from groundgraph.infrastructure.postgres.session import PostgresSession, get_session_factory
from groundgraph.infrastructure.postgres.vector_retriever import PostgresVectorRetriever
from groundgraph.workflows.query_graph import QueryWorkflow, QueryWorkflowConfig

MAX_REQUEST_ID_LENGTH = 128


class Identity(BaseModel):
    """Trusted identity extracted from the authenticated request context.

    In production this would be populated from a verified JWT, mTLS client cert,
    or similar trusted auth mechanism.  The current implementation extracts from
    well-known request headers set by a trusted upstream gateway.
    """

    tenant_id: str
    principal: str


def get_identity(
    request: Request,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    x_principal: str | None = Header(default=None, alias="X-Principal"),
) -> Identity:
    """Extract trusted identity from request headers.

    In production, replace this with a real auth mechanism (JWT verification,
    mTLS identity, etc.).  The header-based approach is acceptable as a
    development/trusted-gateway pattern where the API gateway is the only
    component that populates these headers.
    """
    if not x_tenant_id:
        raise ValueError("X-Tenant-ID header is required")
    if not x_principal:
        raise ValueError("X-Principal header is required")
    return Identity(tenant_id=x_tenant_id, principal=x_principal)


def request_id_from_headers(headers: list[tuple[bytes, bytes]]) -> str:
    """Return a validated correlation ID from an ASGI header list or create one."""

    values = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in headers
        if key.lower() in {b"x-request-id", b"x-correlation-id"}
    }
    value = values.get("x-request-id") or values.get("x-correlation-id")
    if (
        value
        and 1 <= len(value) <= MAX_REQUEST_ID_LENGTH
        and value.isascii()
        and value.replace("-", "").replace("_", "").isalnum()
    ):
        return value
    return f"req-{secrets.token_hex(12)}"


class PostgresHealthChecker:
    name = "postgres"

    async def check(self) -> DependencyHealth:
        conn: Any | None = None
        try:
            async with asyncio.timeout(2):
                conn = cast(
                    Any,
                    await asyncpg.connect(  # pyright: ignore[reportUnknownMemberType]
                        host=self._host,
                        port=self._port,
                        user=self._user,
                        password=self._password,
                        database=self._database,
                    ),
                )
                assert conn is not None
                await conn.execute("SELECT 1")
            return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)
        except TimeoutError:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.TIMEOUT,
            )
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
            )
        finally:
            if conn is not None:
                with suppress(Exception):
                    await conn.close()

    def __init__(self, host: str, port: int, user: str, password: str, database: str) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database


class Neo4jHealthChecker:
    name = "neo4j"

    async def check(self) -> DependencyHealth:
        try:
            driver = AsyncGraphDatabase.driver(  # pyright: ignore[reportUnknownMemberType]
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=1,
                connection_timeout=2.0,
            )
            try:
                async with driver.session() as session:  # pyright: ignore[reportUnknownMemberType]
                    await session.run("RETURN 1")
            finally:
                await driver.close()
            return DependencyHealth(name=self.name, healthy=True, reason_code=HealthReasonCode.OK)
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
            )

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password


class MinioHealthChecker:
    name = "minio"

    async def check(self) -> DependencyHealth:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._endpoint}/minio/health/ready")
                if resp.status_code == 200:  # noqa: PLR2004
                    return DependencyHealth(
                        name=self.name, healthy=True, reason_code=HealthReasonCode.OK
                    )
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    reason_code=HealthReasonCode.UNHEALTHY,
                )
        except Exception:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                reason_code=HealthReasonCode.UNHEALTHY,
            )

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint


def build_health_service(settings: Settings) -> HealthService:
    """Build real, bounded connectivity checks for local dependencies."""
    return HealthService(
        checkers={
            "postgres": PostgresHealthChecker(
                settings.postgres_host,
                settings.postgres_port,
                settings.postgres_user,
                settings.postgres_password.get_secret_value(),
                settings.postgres_db,
            ),
            "neo4j": Neo4jHealthChecker(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password.get_secret_value(),
            ),
            "minio": MinioHealthChecker(settings.s3_endpoint_url),
        }
    )


def _build_retrieval_service(settings: Settings) -> RetrievalService:
    session_factory = get_session_factory()
    embedding_provider = OpenAIEmbeddingProvider(settings=settings)
    vector_retriever = PgVectorContentRetriever(PostgresVectorRetriever(session_factory))
    keyword_retriever = PgKeywordRetrieverAdapter(PostgresKeywordRetriever(session_factory))
    reranker = CrossEncoderReranker(settings=settings)
    answer_generator = EvidenceOnlyAnswerGenerator(settings=settings)
    index_version_resolver = PostgresIndexVersionResolver(session_factory)
    return RetrievalService(
        RetrievalServiceConfig(
            session_factory=session_factory,
            embedding_provider=embedding_provider,
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            reranker=reranker,
            answer_generator=answer_generator,
            index_version_resolver=index_version_resolver,
            settings=settings,
        )
    )


def get_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    """Return a cached RetrievalService instance for the request."""
    return _build_retrieval_service(settings)


def _build_query_workflow(settings: Settings) -> QueryWorkflow:
    session_factory = get_session_factory()
    embedding_provider = OpenAIEmbeddingProvider(settings=settings)
    vector_retriever = PgVectorContentRetriever(PostgresVectorRetriever(session_factory))
    keyword_retriever = PgKeywordRetrieverAdapter(PostgresKeywordRetriever(session_factory))
    reranker = CrossEncoderReranker(settings=settings)
    answer_generator = EvidenceOnlyAnswerGenerator(settings=settings)
    index_version_resolver = PostgresIndexVersionResolver(session_factory)
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
    )
    neo4j_repo = Neo4jGraphRepository(driver=neo4j_driver, database=settings.neo4j_database)

    entity_extractor = LLMEntityExtractor(settings=settings)
    entity_resolver = EntityResolutionService(graph_repository=neo4j_repo)
    planner = RetrievalPlanner(
        RetrievalPlannerConfig(
            entity_extractor=entity_extractor,
            entity_resolver=entity_resolver,
            embedding_provider=embedding_provider,
        )
    )

    return QueryWorkflow(
        config=QueryWorkflowConfig(
            session_factory=session_factory,
            planner=planner,
            embedding_provider=embedding_provider,
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            graph_repository=neo4j_repo,
            reranker=reranker,
            answer_generator=answer_generator,
            index_version_resolver=index_version_resolver,
            settings=settings,
        )
    )


def get_execution_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExecutionRepository:
    """Return an ExecutionRepository backed by a fresh session."""
    factory = get_session_factory()
    session = factory()
    return ExecutionRepository(cast(PostgresSession, session))


async def get_execution_repo_with_session(
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Yield an ExecutionRepository with a session that commits on success.

    The session is always closed on exit. Use this for API endpoints that need
    atomic read-write access to execution runs.
    """
    factory = get_session_factory()
    async with factory() as session:
        repo = ExecutionRepository(cast(PostgresSession, session))
        try:
            yield repo
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_query_workflow(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryWorkflow:
    """Return a cached QueryWorkflow instance for the request."""
    return _build_query_workflow(settings)
