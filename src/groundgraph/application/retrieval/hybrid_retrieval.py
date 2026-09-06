"""Hybrid retrieval service combining vector + keyword + graph retrieval.

Application-layer service that orchestrates hybrid evidence fusion using
RRF for vector+keyword and graph+vector combination. All cross-tenant
access is blocked at the SQL and graph layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID, uuid4

from groundgraph.application.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    EvidenceReranker,
    GraphRepository,
    IndexVersionResolver,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.fusion import reciprocal_rank_fusion
from groundgraph.application.retrieval.graph_fusion import GraphFusionService, hybrid_rrf_fusion
from groundgraph.application.retrieval.retrieval_planner import RetrievalPlanner
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.knowledge import CanonicalEntity
from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan


@dataclass
class HybridRetrievalResult:
    response: QueryResponse
    evidence: list[Evidence]


@dataclass
class HybridRetrievalConfig:
    session_factory: Any
    embedding_provider: EmbeddingProvider
    vector_retriever: VectorContentRetriever
    keyword_retriever: KeywordRetrieverPort
    graph_repository: GraphRepository
    reranker: EvidenceReranker
    answer_generator: AnswerGenerator
    index_version_resolver: IndexVersionResolver
    planner: RetrievalPlanner
    settings: Settings | None = None


class HybridRetrievalService:
    def __init__(self, config: HybridRetrievalConfig) -> None:
        self._session_factory = config.session_factory
        self._embed = config.embedding_provider
        self._vector_retriever = config.vector_retriever
        self._keyword_retriever = config.keyword_retriever
        self._graph_repo = config.graph_repository
        self._reranker = config.reranker
        self._answer_generator = config.answer_generator
        self._index_version_resolver = config.index_version_resolver
        self._planner = config.planner
        self._settings = config.settings or get_settings()

    async def query(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        index_name: str | None = None,
        vector_top_k: int | None = None,
        final_limit: int | None = None,
    ) -> QueryResponse:
        if not tenant_id:
            raise ValueError("tenant_id is required for retrieval")

        index_name = index_name or self._settings.default_index_name
        vector_top_k = vector_top_k or self._settings.vector_top_k
        final_limit = final_limit or self._settings.final_evidence_limit

        plan = await self._planner.plan(question, principal, tenant_id)

        active_index = await self._index_version_resolver.resolve_active(index_name)
        if active_index is None:
            raise RuntimeError(f"No active IndexVersion for index_name={index_name!r}")
        if active_index.embedding_model != self._embed.model:
            raise ValueError(
                f"Embedding model mismatch: index={active_index.embedding_model}, "
                f"provider={self._embed.model}"
            )

        index_version_id = active_index.version_id
        query_vector = await self._embed.embed_one(question)

        vector_results = await self._vector_retriever.search(
            query_vector,
            vector_top_k,
            allowed_principals=[principal],
            tenant_id=tenant_id,
            index_version_id=index_version_id,
        )

        keyword_results = await self._keyword_retriever.search(
            question,
            self._settings.keyword_top_k,
            allowed_principals=[principal],
            tenant_id=tenant_id,
        )

        all_evidence: list[Evidence] = []
        graph_evidence: list[Evidence] = []

        if plan.strategy in ("graph", "hybrid") and plan.entities:
            graph_evidence = await self._retrieve_graph_evidence(plan, principal, tenant_id)

        if plan.strategy == "hybrid":
            fused = self._hybrid_fuse(vector_results, keyword_results, graph_evidence)
            all_evidence = fused[:final_limit]
        elif plan.strategy == "graph":
            all_evidence = graph_evidence[:final_limit]
        else:
            fused_results = reciprocal_rank_fusion(vector_results, keyword_results)
            all_evidence = []
            for r in fused_results[:final_limit]:
                retrieval_method: Literal["vector", "keyword"] = (
                    "vector" if r.vector_score else "keyword"
                )
                all_evidence.append(
                    Evidence(
                        evidence_id=uuid4(),
                        source_id=r.source_id,
                        document_id=r.document_id,
                        version_id=r.version_id,
                        chunk_id=r.chunk_id,
                        content=r.content,
                        retrieval_method=retrieval_method,
                        vector_score=r.vector_score,
                        allowed_principals=list(r.allowed_principals),
                    )
                )

        deduplicated = self._deduplicate(all_evidence)
        reranked = await self._reranker.rerank(deduplicated, question)

        return await self._answer_generator.generate(question, reranked, plan)

    async def _retrieve_graph_evidence(
        self, plan: RetrievalPlan, principal: str, tenant_id: str
    ) -> list[Evidence]:
        graph_fusion_svc = GraphFusionService(graph_repository=self._graph_repo)
        entities = await self._resolve_entities(plan)
        return await graph_fusion_svc.retrieve_evidence(
            entities,
            predicates=plan.predicates,
            valid_at=plan.valid_at,
            max_depth=plan.max_graph_depth,
            tenant_id=tenant_id,
            principal=principal,
        )

    async def _resolve_entities(self, plan: RetrievalPlan) -> list[CanonicalEntity]:
        resolved: list[CanonicalEntity] = []
        for re in plan.entities:
            entity = await self._graph_repo.get_entity(re.entity_id)
            if entity is not None:
                resolved.append(entity)
        return resolved

    def _hybrid_fuse(
        self,
        vector_results: list[Any],
        keyword_results: list[Any],
        graph_evidence: list[Evidence],
    ) -> list[Evidence]:
        return hybrid_rrf_fusion(vector_results, keyword_results, graph_evidence)

    def _deduplicate(self, evidence: list[Evidence]) -> list[Evidence]:
        seen: set[UUID] = set()
        unique: list[Evidence] = []
        for e in evidence:
            if e.evidence_id not in seen:
                seen.add(e.evidence_id)
                unique.append(e)
        return unique

    async def query_with_evidence(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        *,
        index_name: str | None = None,
        vector_top_k: int | None = None,
        final_limit: int | None = None,
    ) -> HybridRetrievalResult:
        """Execute hybrid retrieval and return response alongside the evidence used.

        This method is intended for workflow use where the evidence list must be
        captured for downstream claim validation.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for retrieval")

        index_name = index_name or self._settings.default_index_name
        vector_top_k = vector_top_k or self._settings.vector_top_k
        final_limit = final_limit or self._settings.final_evidence_limit

        plan = await self._planner.plan(question, principal, tenant_id)

        active_index = await self._index_version_resolver.resolve_active(index_name)
        if active_index is None:
            raise RuntimeError(f"No active IndexVersion for index_name={index_name!r}")
        if active_index.embedding_model != self._embed.model:
            raise ValueError(
                f"Embedding model mismatch: index={active_index.embedding_model}, "
                f"provider={self._embed.model}"
            )

        index_version_id = active_index.version_id
        query_vector = await self._embed.embed_one(question)

        vector_results = await self._vector_retriever.search(
            query_vector,
            vector_top_k,
            allowed_principals=[principal],
            tenant_id=tenant_id,
            index_version_id=index_version_id,
        )

        keyword_results = await self._keyword_retriever.search(
            question,
            self._settings.keyword_top_k,
            allowed_principals=[principal],
            tenant_id=tenant_id,
        )

        all_evidence: list[Evidence] = []
        graph_evidence: list[Evidence] = []

        if plan.strategy in ("graph", "hybrid") and plan.entities:
            graph_evidence = await self._retrieve_graph_evidence(plan, principal, tenant_id)

        if plan.strategy == "hybrid":
            fused = self._hybrid_fuse(vector_results, keyword_results, graph_evidence)
            all_evidence = fused[:final_limit]
        elif plan.strategy == "graph":
            all_evidence = graph_evidence[:final_limit]
        else:
            fused_results = reciprocal_rank_fusion(vector_results, keyword_results)
            all_evidence = []
            for r in fused_results[:final_limit]:
                retrieval_method: Literal["vector", "keyword"] = (
                    "vector" if r.vector_score else "keyword"
                )
                all_evidence.append(
                    Evidence(
                        evidence_id=uuid4(),
                        source_id=r.source_id,
                        document_id=r.document_id,
                        version_id=r.version_id,
                        chunk_id=r.chunk_id,
                        content=r.content,
                        retrieval_method=retrieval_method,
                        vector_score=r.vector_score,
                        allowed_principals=list(r.allowed_principals),
                    )
                )

        deduplicated = self._deduplicate(all_evidence)
        reranked = await self._reranker.rerank(deduplicated, question)
        response = await self._answer_generator.generate(question, reranked, plan)
        return HybridRetrievalResult(response=response, evidence=reranked)
