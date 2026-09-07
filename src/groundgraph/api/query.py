"""Query endpoints: vector (M4 baseline) and hybrid (M7)."""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from groundgraph.api.dependencies import (
    Identity,
    get_execution_repo_with_session,
    get_identity,
    get_query_workflow,
    get_retrieval_service,
)
from groundgraph.application.retrieval.retrieval_service import RetrievalService
from groundgraph.domain.execution import ExecutionRun, ExecutionRunStatus
from groundgraph.domain.retrieval import QueryResponse
from groundgraph.workflows.query_graph import QueryWorkflow

router = APIRouter(tags=["query"])
v1_router = APIRouter(prefix="/v1", tags=["v1-query"])

LOG = logging.getLogger(__name__)


class VectorQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(..., min_length=1)
    principal_id: str = Field(..., min_length=1)
    index_name: str | None = None
    vector_top_k: int | None = Field(default=None, ge=1, le=100)
    final_limit: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    citation_id: UUID
    claim_id: UUID
    evidence_id: UUID
    locator: str


class ClaimModel(BaseModel):
    claim_id: UUID
    text: str
    factual: bool
    support_status: str
    evidence_ids: list[UUID]


class VectorQueryResponse(BaseModel):
    answer: str | None
    status: str
    claims: list[ClaimModel]
    citations: list[CitationModel]
    confidence_band: str
    execution_run_id: UUID
    warnings: list[str] = Field(default_factory=list)


class HybridQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(..., min_length=1)
    principal_id: str = Field(..., min_length=1)
    index_name: str | None = None


class HybridQueryTrustedRequest(BaseModel):
    """Request body for the trusted /v1/query endpoint.

    tenant_id and principal are extracted from the trusted request identity
    (X-Tenant-ID / X-Principal headers), not from this body.
    """

    question: str = Field(..., min_length=1, max_length=2000)
    index_name: str | None = None


def _handle_error(exc: Exception, request_id: str) -> None:
    """Log the internal error and re-raise as HTTPException with safe detail."""
    LOG.exception("Query failed [request_id=%s]: %s", request_id, exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
        headers={"X-Request-ID": request_id},
    ) from exc


@router.post("/vector", response_model=VectorQueryResponse)
async def query_vector(
    request: VectorQueryRequest,
    svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
    http_request: Request,
) -> VectorQueryResponse:
    """Execute a vector-only query and return an evidence-grounded answer.

    This is the M4 baseline vertical slice: embed → vector search → keyword search
    → RRF → rerank → evidence-only answer → citations.
    """
    request_id = http_request.headers.get("x-request-id", f"req-{secrets.token_hex(12)}")
    result: QueryResponse | None = None
    try:
        result = await svc.query(
            question=request.question,
            principal=request.principal_id,
            tenant_id=request.tenant_id,
            index_name=request.index_name,
            vector_top_k=request.vector_top_k,
            final_limit=request.final_limit,
        )
    except Exception as exc:
        _handle_error(exc, request_id)

    assert result is not None

    assert result is not None
    return VectorQueryResponse(
        answer=result.answer,
        status=result.status,
        claims=[
            ClaimModel(
                claim_id=c.claim_id,
                text=c.text,
                factual=c.factual,
                support_status=c.support_status,
                evidence_ids=c.evidence_ids,
            )
            for c in result.claims
        ],
        citations=[
            CitationModel(
                citation_id=cit.citation_id,
                claim_id=cit.claim_id,
                evidence_id=cit.evidence_id,
                locator=cit.locator,
            )
            for cit in result.citations
        ],
        confidence_band=result.confidence_band,
        execution_run_id=result.execution_run_id,
        warnings=result.warnings,
    )


@router.post("/hybrid", response_model=VectorQueryResponse)
async def query_hybrid(
    request: HybridQueryRequest,
    workflow: Annotated[QueryWorkflow, Depends(get_query_workflow)],
    http_request: Request,
) -> VectorQueryResponse:
    """Execute a hybrid GraphRAG query using the full query workflow.

    This is the M7 endpoint: hybrid vector + keyword + graph retrieval
    with claim validation and fail-closed responses when evidence is insufficient.
    """
    request_id = http_request.headers.get("x-request-id", f"req-{secrets.token_hex(12)}")
    result: QueryResponse | None = None
    try:
        result = await workflow.ainvoke(
            question=request.question,
            principal=request.principal_id,
            tenant_id=request.tenant_id,
            index_name=request.index_name,
        )
    except Exception as exc:
        _handle_error(exc, request_id)

    assert result is not None

    return VectorQueryResponse(
        answer=result.answer,
        status=result.status,
        claims=[
            ClaimModel(
                claim_id=c.claim_id,
                text=c.text,
                factual=c.factual,
                support_status=c.support_status,
                evidence_ids=c.evidence_ids,
            )
            for c in result.claims
        ],
        citations=[
            CitationModel(
                citation_id=cit.citation_id,
                claim_id=cit.claim_id,
                evidence_id=cit.evidence_id,
                locator=cit.locator,
            )
            for cit in result.citations
        ],
        confidence_band=result.confidence_band,
        execution_run_id=result.execution_run_id,
        warnings=result.warnings,
    )


@v1_router.post("/query", response_model=VectorQueryResponse, name="query-hybrid-trusted")
async def query_v1(
    request: HybridQueryTrustedRequest,
    workflow: Annotated[QueryWorkflow, Depends(get_query_workflow)],
    identity: Annotated[Identity, Depends(get_identity)],
    http_request: Request,
    repo: Annotated[object, Depends(get_execution_repo_with_session)],
) -> VectorQueryResponse:
    """Execute a hybrid GraphRAG query using trusted identity from request context.

    This is the M7 stable endpoint. tenant_id and principal are extracted from
    trusted request identity (X-Tenant-ID / X-Principal headers set by gateway),
    NOT from client-supplied body fields.

    Execution runs are persisted for audit and replay.
    """
    request_id = http_request.headers.get("x-request-id", f"req-{secrets.token_hex(12)}")

    run_input: dict[str, object] = {"question": request.question}
    if request.index_name:
        run_input["index_name"] = request.index_name

    run = ExecutionRun(
        run_id=uuid4(),
        workflow="query",
        status=ExecutionRunStatus.PENDING,
        principal=identity.principal,
        tenant_id=identity.tenant_id,
        input=run_input,
        output={},
        started_at=datetime.now(UTC),
    )
    await repo.create_run(run, commit=True)  # type: ignore[attr-defined]

    result: QueryResponse | None = None
    try:
        await repo.update_run_status(  # type: ignore[attr-defined]
            run_id=run.run_id,
            expected_status=ExecutionRunStatus.PENDING,
            new_status=ExecutionRunStatus.RUNNING,
            commit=True,
        )

        result = await workflow.ainvoke(
            question=request.question,
            principal=identity.principal,
            tenant_id=identity.tenant_id,
            index_name=request.index_name,
        )

        run_output: dict[str, object] = {
            "answer": result.answer,
            "status": result.status,
            "claims_count": len(result.claims),
            "citation_ids": [c.evidence_id for c in result.citations],
            "confidence_band": result.confidence_band,
            "warnings": result.warnings,
        }
        await repo.update_run_status(  # type: ignore[attr-defined]
            run_id=run.run_id,
            expected_status=ExecutionRunStatus.RUNNING,
            new_status=ExecutionRunStatus.SUCCEEDED,
            output=run_output,
            commit=True,
        )
    except Exception as exc:
        await repo.update_run_status(  # type: ignore[attr-defined]
            run_id=run.run_id,
            expected_status=ExecutionRunStatus.RUNNING,
            new_status=ExecutionRunStatus.FAILED,
            error_code="WORKFLOW_FAILED",
            error_message=str(exc),
            commit=True,
        )
        _handle_error(exc, request_id)

    assert result is not None

    return VectorQueryResponse(
        answer=result.answer,
        status=result.status,
        claims=[
            ClaimModel(
                claim_id=c.claim_id,
                text=c.text,
                factual=c.factual,
                support_status=c.support_status,
                evidence_ids=c.evidence_ids,
            )
            for c in result.claims
        ],
        citations=[
            CitationModel(
                citation_id=cit.citation_id,
                claim_id=cit.claim_id,
                evidence_id=cit.evidence_id,
                locator=cit.locator,
            )
            for cit in result.citations
        ],
        confidence_band=result.confidence_band,
        execution_run_id=run.run_id,
        warnings=result.warnings,
    )
