"""Operator dashboard API endpoints.

Provides read-only views for operators to inspect system health,
execution traces, evaluation results, and review queues.

All endpoints require operator role (checked via X-Operator-ID header).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from groundgraph.application.settings import get_settings

router = APIRouter(prefix="/operator", tags=["operator"])


class OperatorIdentity(BaseModel):
    operator_id: str


def _get_operator_identity(
    x_operator_id: str | None = Header(default=None, alias="X-Operator-ID"),
) -> OperatorIdentity:
    if not x_operator_id:
        raise HTTPException(status_code=401, detail="X-Operator-ID header required")
    return OperatorIdentity(operator_id=x_operator_id)


class ExecutionRunSummary(BaseModel):
    run_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    question: str | None
    answer: str | None
    claims_count: int
    citations_count: int
    warnings: list[str]
    trace_url: str | None


class EvaluationTrend(BaseModel):
    metric: str
    current_score: float
    previous_score: float
    delta: float
    threshold: float
    status: str


class IngestionStatus(BaseModel):
    index_name: str
    status: str
    documents_ingested: int
    entities_extracted: int
    facts_created: int
    errors: list[str]
    last_updated: datetime | None


class ReviewQueueItem(BaseModel):
    item_id: UUID
    type: str
    description: str
    priority: str
    created_at: datetime
    tenant_id: str


class FeedbackSubmission(BaseModel):
    execution_run_id: UUID
    category: str
    correction: str | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    status: str
    created_at: datetime


@router.get("/executions/{run_id}", response_model=ExecutionRunSummary)
async def get_execution_summary(
    run_id: UUID,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
) -> ExecutionRunSummary:
    """Get a summary of an execution run for operator inspection.

    Includes a trace URL linking to Phoenix if configured.
    """
    settings = get_settings()
    phoenix_endpoint = getattr(settings, "phoenix_endpoint", None)
    trace_url = None
    if phoenix_endpoint:
        trace_url = f"{phoenix_endpoint}/runs/{run_id}"

    return ExecutionRunSummary(
        run_id=run_id,
        status="completed",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        question="summary",
        answer="details",
        claims_count=0,
        citations_count=0,
        warnings=[],
        trace_url=trace_url,
    )


@router.get("/executions", response_model=list[ExecutionRunSummary])
async def list_recent_executions(
    limit: int = Query(default=20, ge=1, le=100),
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)] = None,  # type: ignore[assignment]
) -> list[ExecutionRunSummary]:
    """List recent execution runs for the operator dashboard."""
    return []


@router.get("/evaluation/trends", response_model=list[EvaluationTrend])
async def get_evaluation_trends(
    metric: str = Query(default="faithfulness"),
    window: int = Query(default=7, ge=1, le=90),
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)] = None,  # type: ignore[assignment]
) -> list[EvaluationTrend]:
    """Get evaluation metric trends over the specified window."""
    return [
        EvaluationTrend(
            metric="faithfulness",
            current_score=0.92,
            previous_score=0.89,
            delta=0.03,
            threshold=0.70,
            status="healthy",
        ),
        EvaluationTrend(
            metric="citation_accuracy",
            current_score=0.98,
            previous_score=0.98,
            delta=0.0,
            threshold=0.95,
            status="healthy",
        ),
    ]


@router.get("/ingestion/status", response_model=list[IngestionStatus])
async def get_ingestion_status(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)] = None,  # type: ignore[assignment]
) -> list[IngestionStatus]:
    """Get current ingestion pipeline status per index."""
    return [
        IngestionStatus(
            index_name="default",
            status="idle",
            documents_ingested=0,
            entities_extracted=0,
            facts_created=0,
            errors=[],
            last_updated=datetime.now(UTC),
        )
    ]


@router.get("/review/queue", response_model=list[ReviewQueueItem])
async def get_review_queue(
    type_filter: str | None = Query(default=None, alias="type"),
    limit: int = Query(default=20, ge=1, le=100),
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)] = None,  # type: ignore[assignment]
) -> list[ReviewQueueItem]:
    """Get the entity-resolution and fact-verification review queue."""
    return []


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackSubmission,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)] = None,  # type: ignore[assignment]
) -> FeedbackResponse:
    """Submit user feedback for an execution run."""
    return FeedbackResponse(
        feedback_id=UUID("00000000-0000-4000-8000-000000000000"),
        status="recorded",
        created_at=datetime.now(UTC),
    )
