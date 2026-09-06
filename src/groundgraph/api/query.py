"""M4 vector query endpoint — thin vertical slice."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from groundgraph.api.dependencies import get_retrieval_service
from groundgraph.application.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/query", tags=["query"])


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


@router.post("/vector", response_model=VectorQueryResponse)
async def query_vector(
    request: VectorQueryRequest,
    svc: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> VectorQueryResponse:
    """Execute a vector-only query and return an evidence-grounded answer.

    This is the M4 baseline vertical slice: embed → vector search → keyword search
    → RRF → rerank → evidence-only answer → citations.
    """
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

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
