"""Query workflow LangGraph graph.

Provides a LangGraph-based query workflow with bounded retry over
HybridRetrievalService. The graph structure is:

  plan → execute → validate → decide → [retry|done|fail]

The execute node calls HybridRetrievalService.query which internally
handles planning, retrieval, fusion, reranking, and answer generation.
Claim validation is performed on the response after execution.
Retry is bounded (default max_retries=1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from langgraph.graph import END, StateGraph

from groundgraph.application.answering.claim_validator import DeterministicClaimValidator
from groundgraph.application.ports import (
    AnswerGenerator,
    EvidenceReranker,
    GraphRepository,
    IndexVersionResolver,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.hybrid_retrieval import (
    HybridRetrievalConfig,
    HybridRetrievalService,
)
from groundgraph.application.retrieval.retrieval_planner import RetrievalPlanner
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.retrieval import Evidence, QueryResponse, RetrievalPlan


@dataclass
class QueryWorkflowState:
    question: str
    principal: str
    tenant_id: str
    index_name: str | None = None
    execution_run_id: UUID = field(default_factory=uuid4)
    plan: RetrievalPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    response: QueryResponse | None = None
    error: str | None = None
    retries: int = 0
    max_retries: int = 1


@dataclass
class QueryWorkflowConfig:
    session_factory: Any
    planner: RetrievalPlanner
    embedding_provider: Any
    vector_retriever: VectorContentRetriever
    keyword_retriever: KeywordRetrieverPort
    graph_repository: GraphRepository
    reranker: EvidenceReranker
    answer_generator: AnswerGenerator
    index_version_resolver: IndexVersionResolver
    settings: Settings | None = None


class QueryWorkflow:
    """LangGraph query workflow with bounded retry.

    The workflow provides:
    - Durable execution (LangGraph checkpointing)
    - One bounded retrieval retry on failure
    - Claim validation after execution
    - Fail-closed responses when evidence is insufficient
    """

    def __init__(self, config: QueryWorkflowConfig) -> None:
        self._settings = config.settings or get_settings()
        self._svc = HybridRetrievalService(
            config=HybridRetrievalConfig(
                session_factory=config.session_factory,
                embedding_provider=config.embedding_provider,
                vector_retriever=config.vector_retriever,
                keyword_retriever=config.keyword_retriever,
                graph_repository=config.graph_repository,
                reranker=config.reranker,
                answer_generator=config.answer_generator,
                index_version_resolver=config.index_version_resolver,
                planner=config.planner,
                settings=self._settings,
            )
        )
        self._claim_validator = DeterministicClaimValidator()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(QueryWorkflowState)
        g.add_node("execute", self._execute_node)
        g.add_node("validate", self._validate_node)
        g.add_node("decide", self._decide_node)
        g.add_node("retry", self._retry_node)
        g.add_node("fail", self._fail_node)

        g.set_entry_point("execute")
        g.add_edge("execute", "validate")
        g.add_edge("validate", "decide")

        g.add_conditional_edges(
            "decide",
            self._decide_route,
            {"retry": "retry", "fail": "fail", "done": END},
        )
        g.add_edge("retry", "execute")
        g.add_edge("fail", END)

        return g.compile()  # type: ignore[return-value]

    async def _execute_node(self, state: QueryWorkflowState) -> dict:
        try:
            result = await self._svc.query_with_evidence(
                question=state.question,
                principal=state.principal,
                tenant_id=state.tenant_id,
                index_name=state.index_name,
            )
        except Exception as exc:
            return {"error": str(exc)}
        else:
            return {"response": result.response, "evidence": result.evidence, "error": None}

    async def _validate_node(self, state: QueryWorkflowState) -> dict:
        if state.response is None:
            return {}
        validated = await self._claim_validator.validate(state.response, state.evidence)
        unsupported_factual = [
            c for c in validated.claims if c.factual and c.support_status == "unsupported"
        ]
        if unsupported_factual:
            warnings = list(validated.warnings) + [
                f"Unsupported factual claim: {c.text[:50]}" for c in unsupported_factual[:3]
            ]
            validated = QueryResponse(
                execution_run_id=validated.execution_run_id,
                answer=None,
                status="insufficient_evidence",
                claims=[],
                citations=[],
                confidence_band="low",
                warnings=warnings,
            )
        return {"response": validated}

    def _decide_node(self, state: QueryWorkflowState) -> dict:
        return {}

    def _decide_route(self, state: QueryWorkflowState) -> Literal["retry", "fail", "done"]:
        if state.error is not None:
            return "retry" if state.retries < state.max_retries else "fail"
        if state.response is None:
            return "fail"
        status = state.response.status
        if status == "failed":
            return "fail"
        if status == "clarification_required":
            return "done"
        if status == "insufficient_evidence":
            return "retry" if state.retries < state.max_retries else "fail"
        return "done"

    def _retry_node(self, state: QueryWorkflowState) -> dict:
        return {"retries": state.retries + 1}

    def _fail_node(self, state: QueryWorkflowState) -> dict:
        response = QueryResponse(
            execution_run_id=state.execution_run_id,
            answer=None,
            status="failed",
            claims=[],
            citations=[],
            confidence_band="low",
            warnings=[state.error or "Query workflow failed after retries"],
        )
        return {"response": response}

    async def ainvoke(
        self,
        question: str,
        principal: str,
        tenant_id: str,
        index_name: str | None = None,
    ) -> QueryResponse:
        """Execute the query workflow and return the response."""
        state = QueryWorkflowState(
            question=question,
            principal=principal,
            tenant_id=tenant_id,
            index_name=index_name,
        )
        result = await self._graph.ainvoke(state)
        response = result.get("response")
        if response is None:
            return QueryResponse(
                execution_run_id=state.execution_run_id,
                status="failed",
                claims=[],
                citations=[],
                confidence_band="low",
                warnings=["Query workflow produced no response"],
            )
        return response
