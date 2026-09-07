"""Operator dashboard API endpoints.

Provides read-only views for operators to inspect system health,
execution traces, evaluation results, and review queues.

All endpoints require a verified JWT with operator or admin role claim.
Operator identity is extracted from the Authorization: Bearer token.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from jose import jwt
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from groundgraph.api.dependencies import (
    _get_identity_from_jwt,
    get_execution_repo_with_session,
    get_settings,
)
from groundgraph.application.settings import Settings
from groundgraph.infrastructure.postgres.models import (
    EvaluationResult,
    EvaluationRun,
    ExecutionRun,
    HumanReviewItem,
    ImprovementProposal,
    Source,
    UserFeedback,
)
from groundgraph.infrastructure.postgres.session import get_session_factory

router = APIRouter(prefix="/operator", tags=["operator"])


ALLOWED_OPERATOR_ROLES = {"operator", "admin"}


class OperatorIdentity(BaseModel):
    operator_id: str
    tenant_id: str
    is_global: bool


def _get_operator_identity(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    settings: Settings = Depends(get_settings),
) -> OperatorIdentity:
    """Verify JWT and extract operator identity.

    Requires a valid JWT with operator or admin role.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    token = authorization[7:]

    if settings.auth_mode == "local":
        if not settings.auth_trusted_headers:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Header auth not allowed in this environment",
            )
        try:
            unverified_claims = jwt.get_unverified_claims(token)
            claims = dict(unverified_claims)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWT format",
            ) from exc
        role = claims.get("role", "")
        if isinstance(role, list):
            has_role = bool(set(role) & ALLOWED_OPERATOR_ROLES)
        else:
            has_role = role in ALLOWED_OPERATOR_ROLES
        if not has_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator or admin role required",
            )
        tenant_id = str(claims.get("tenant_id", settings.auth_default_tenant))
        principal = str(claims.get("sub", "unknown"))
        is_global = claims.get("is_global_operator", False) is True
        return OperatorIdentity(
            operator_id=principal,
            tenant_id=tenant_id,
            is_global=is_global,
        )

    identity = _get_identity_from_jwt(token, settings)
    claims = {}
    try:
        unverified = jwt.get_unverified_claims(token)
        claims = dict(unverified)
    except Exception:
        pass

    role = claims.get("role", "")
    if isinstance(role, list):
        has_role = bool(set(role) & ALLOWED_OPERATOR_ROLES)
    else:
        has_role = role in ALLOWED_OPERATOR_ROLES

    if not has_role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator or admin role required",
        )

    is_global = claims.get("is_global_operator", False) is True

    return OperatorIdentity(
        operator_id=identity.principal,
        tenant_id=identity.tenant_id,
        is_global=is_global,
    )


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
    vote: str = Field(pattern="^(thumbs_up|thumbs_down)$")
    category: str | None = None
    correction: str | None = None
    notes: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    status: str
    created_at: datetime


class ReviewDecisionSubmission(BaseModel):
    review_id: UUID
    decision: str = Field(pattern="^(approved|rejected)$")
    notes: str | None = None


class ReviewDecisionResponse(BaseModel):
    review_id: UUID
    decision: str
    decided_by: str
    decided_at: datetime


@router.get("/executions/{run_id}", response_model=ExecutionRunSummary)
async def get_execution_summary(
    run_id: UUID,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    repo: Annotated[object, Depends(get_execution_repo_with_session)],
) -> ExecutionRunSummary:
    """Get a summary of an execution run for operator inspection."""
    settings = get_settings()
    phoenix_endpoint = getattr(settings, "phoenix_endpoint", None)
    trace_url = f"{phoenix_endpoint}/runs/{run_id}" if phoenix_endpoint else None

    try:
        run = await repo.get_run(run_id)  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        ) from exc

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {run_id} not found",
        )

    if not operator.is_global and run.tenant_id != operator.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access executions from other tenants",
        )

    question = run.input.get("question") if run.input else None
    answer = run.output.get("answer") if run.output else None
    warnings = run.output.get("warnings", []) if run.output else []
    claims_count = len(run.output.get("claims", [])) if run.output else 0
    citations_count = len(run.output.get("citations", [])) if run.output else 0

    return ExecutionRunSummary(
        run_id=run.run_id,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        question=question,
        answer=answer,
        claims_count=claims_count,
        citations_count=citations_count,
        warnings=warnings,
        trace_url=trace_url,
    )


@router.get("/executions", response_model=list[ExecutionRunSummary])
async def list_recent_executions(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    repo: Annotated[object, Depends(get_execution_repo_with_session)],
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ExecutionRunSummary]:
    """List recent execution runs for the operator dashboard."""
    settings = get_settings()
    phoenix_endpoint = getattr(settings, "phoenix_endpoint", None)

    try:
        runs = await repo.list_recent_runs(limit=limit)  # type: ignore[attr-defined]
    except Exception:
        return []

    summaries = []
    for run in runs:
        if not operator.is_global and run.tenant_id != operator.tenant_id:
            continue
        q = run.input.get("question") if run.input else None
        a = run.output.get("answer") if run.output else None
        w = run.output.get("warnings", []) if run.output else []
        cc = len(run.output.get("claims", [])) if run.output else 0
        cic = len(run.output.get("citations", [])) if run.output else 0
        trace = f"{phoenix_endpoint}/runs/{run.run_id}" if phoenix_endpoint else None
        summaries.append(
            ExecutionRunSummary(
                run_id=run.run_id,
                status=run.status,
                started_at=run.started_at,
                finished_at=run.finished_at,
                question=q,
                answer=a,
                claims_count=cc,
                citations_count=cic,
                warnings=w,
                trace_url=trace,
            )
        )
    return summaries


@router.get("/evaluation/trends", response_model=list[EvaluationTrend])
async def get_evaluation_trends(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    settings: Annotated[Settings, Depends(get_settings)],
    metric: str = Query(default="faithfulness"),
    window: int = Query(default=7, ge=1, le=90),
) -> list[EvaluationTrend]:
    """Get evaluation metric trends over the specified window."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            cutoff = datetime.now(UTC).timestamp() - (window * 86400)
            stmt = (
                select(EvaluationResult.metrics, EvaluationRun.created_at)
                .join(EvaluationRun, EvaluationRun.run_id == EvaluationResult.run_id)
                .where(EvaluationRun.created_at >= datetime.fromtimestamp(cutoff, tz=UTC))
                .order_by(EvaluationRun.created_at.desc())
                .limit(100)
            )
            result = await session.execute(stmt)
            rows = result.all()
    except Exception:
        rows = []

    if not rows:
        return [
            EvaluationTrend(
                metric="faithfulness",
                current_score=0.0,
                previous_score=0.0,
                delta=0.0,
                threshold=0.70,
                status="no_data",
            ),
            EvaluationTrend(
                metric="citation_accuracy",
                current_score=0.0,
                previous_score=0.0,
                delta=0.0,
                threshold=0.95,
                status="no_data",
            ),
        ]

    f_scores = [
        r.metrics.get("faithfulness") for r in rows if r.metrics.get("faithfulness") is not None
    ]
    c_scores = [
        r.metrics.get("citation_accuracy")
        for r in rows
        if r.metrics.get("citation_accuracy") is not None
    ]

    top_n = 10
    prev_offset = 10
    faith_threshold = 0.70
    cite_threshold = 0.95

    def _avg(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    cur_f = _avg(f_scores[:top_n])
    prev_f = _avg(f_scores[prev_offset : prev_offset * 2]) if len(f_scores) > top_n else cur_f
    cur_c = _avg(c_scores[:top_n])
    prev_c = _avg(c_scores[prev_offset : prev_offset * 2]) if len(c_scores) > top_n else cur_c

    return [
        EvaluationTrend(
            metric="faithfulness",
            current_score=round(cur_f, 3),
            previous_score=round(prev_f, 3),
            delta=round(cur_f - prev_f, 3),
            threshold=faith_threshold,
            status="healthy" if cur_f >= faith_threshold else "degraded",
        ),
        EvaluationTrend(
            metric="citation_accuracy",
            current_score=round(cur_c, 3),
            previous_score=round(prev_c, 3),
            delta=round(cur_c - prev_c, 3),
            threshold=cite_threshold,
            status="healthy" if cur_c >= cite_threshold else "degraded",
        ),
    ]


@router.get("/ingestion/status", response_model=list[IngestionStatus])
async def get_ingestion_status(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[IngestionStatus]:
    """Get current ingestion pipeline status per index."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            tenant_filter = {} if operator.is_global else {"tenant_id": operator.tenant_id}
            stmt = (
                select(Source.tenant_id, func.count(Source.source_id).label("doc_count"))
                .filter_by(**tenant_filter)
                .where(Source.is_active == True)  # noqa: E712
                .group_by(Source.tenant_id)
            )
            result = await session.execute(stmt)
            rows = result.all()
    except Exception:
        rows = []

    if not rows:
        return [
            IngestionStatus(
                index_name="default",
                status="idle",
                documents_ingested=0,
                entities_extracted=0,
                facts_created=0,
                errors=[],
                last_updated=None,
            )
        ]

    return [
        IngestionStatus(
            index_name=row.tenant_id,
            status="active",
            documents_ingested=row.doc_count,
            entities_extracted=0,
            facts_created=0,
            errors=[],
            last_updated=datetime.now(UTC),
        )
        for row in rows
    ]


@router.get("/review/queue", response_model=list[ReviewQueueItem])
async def get_review_queue(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    settings: Annotated[Settings, Depends(get_settings)],
    type_filter: str | None = Query(default=None, alias="type"),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ReviewQueueItem]:
    """Get the entity-resolution and fact-verification review queue."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            tenant_cond = (
                [] if operator.is_global else [ExecutionRun.tenant_id == operator.tenant_id]
            )
            stmt = (
                select(HumanReviewItem, ExecutionRun.tenant_id.label("run_tenant_id"))
                .join(ExecutionRun, ExecutionRun.run_id == HumanReviewItem.run_id, isouter=True)
                .where(*tenant_cond, HumanReviewItem.decision.is_(None))
                .order_by(HumanReviewItem.created_at.desc())
                .limit(limit)
            )
            if type_filter:
                stmt = stmt.where(HumanReviewItem.item_type == type_filter)
            result = await session.execute(stmt)
            rows = result.all()
    except Exception:
        rows = []

    return [
        ReviewQueueItem(
            item_id=review_item.review_id,
            type=review_item.item_type,
            description=f"Review {review_item.item_type}: run={review_item.run_id}",
            priority="normal",
            created_at=review_item.created_at,
            tenant_id=run_tenant_id or operator.tenant_id,
        )
        for review_item, run_tenant_id in rows
    ]


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackSubmission,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FeedbackResponse:
    """Submit user feedback for an execution run.

    Verifies tenant ownership before creating feedback: the execution run
    must belong to the operator's tenant (unless operator is global).
    """
    run_not_found = False
    cross_tenant = False
    feedback_id: UUID | None = None
    created_at: datetime | None = None
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            run_result = await session.execute(
                select(ExecutionRun.tenant_id).where(
                    ExecutionRun.run_id == feedback.execution_run_id
                )
            )
            run_row = run_result.scalar_one_or_none()
            if run_row is None:
                run_not_found = True
            elif not operator.is_global and run_row != operator.tenant_id:
                cross_tenant = True
            else:
                record = UserFeedback(
                    run_id=feedback.execution_run_id,
                    vote=feedback.vote,
                    category=feedback.category,
                    notes=feedback.notes,
                    correction=feedback.correction,
                )
                session.add(record)
                await session.commit()
                feedback_id = record.feedback_id
                created_at = record.created_at
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {exc}",
        ) from exc

    if run_not_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution run {feedback.execution_run_id} not found",
        )
    if cross_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot submit feedback for execution runs from other tenants",
        )

    assert feedback_id is not None
    assert created_at is not None

    return FeedbackResponse(
        feedback_id=feedback_id,
        status="recorded",
        created_at=created_at,
    )


@router.post("/review/decisions", response_model=ReviewDecisionResponse)
async def submit_review_decision(
    decision: ReviewDecisionSubmission,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReviewDecisionResponse:
    """Submit a decision for a review queue item.

    Verifies tenant ownership before updating: the review item's execution run
    must belong to the operator's tenant (unless operator is global).
    """
    not_found = False
    cross_tenant = False
    resolved_at: datetime | None = None
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            row_result = await session.execute(
                select(HumanReviewItem, ExecutionRun.tenant_id.label("run_tenant_id"))
                .join(
                    ExecutionRun,
                    ExecutionRun.run_id == HumanReviewItem.run_id,
                    isouter=True,
                )
                .where(HumanReviewItem.review_id == decision.review_id)
            )
            row = row_result.one_or_none()
            if row is None:
                not_found = True
            else:
                review_item, run_tenant_id = row
                tenant_id = run_tenant_id or getattr(review_item, "tenant_id", None)
                if (tenant_id is None and not operator.is_global) or (
                    not operator.is_global
                    and tenant_id is not None
                    and tenant_id != operator.tenant_id
                ):
                    cross_tenant = True
                else:
                    stmt = (
                        update(HumanReviewItem)
                        .where(HumanReviewItem.review_id == decision.review_id)
                        .values(
                            decision=decision.decision,
                            decided_by=operator.operator_id,
                            decided_at=datetime.now(UTC),
                        )
                    )
                    await session.execute(stmt)
                    await session.commit()
                    result = await session.execute(
                        select(HumanReviewItem.decided_at).where(
                            HumanReviewItem.review_id == decision.review_id
                        )
                    )
                    resolved_row = result.scalar_one_or_none()
                    not_found = resolved_row is None
                    resolved_at = datetime.now(UTC) if not_found else resolved_row
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record decision",
        ) from exc

    if not_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review item {decision.review_id} not found",
        )
    if cross_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify reviews from other tenants",
        )

    assert resolved_at is not None

    return ReviewDecisionResponse(
        review_id=decision.review_id,
        decision=decision.decision,
        decided_by=operator.operator_id,
        decided_at=resolved_at,
    )


PROPOSAL_STATES = {"PROPOSED", "EVALUATED", "APPROVED", "CANARY", "PROMOTED", "ROLLED_BACK"}
VALID_TRANSITIONS = {
    "PROPOSED": {"EVALUATED"},
    "EVALUATED": {"APPROVED"},
    "APPROVED": {"CANARY"},
    "CANARY": {"PROMOTED", "ROLLED_BACK"},
    "PROMOTED": set(),
    "ROLLED_BACK": set(),
}


class ProposalSubmission(BaseModel):
    failure_cluster_id: str | None = None
    baseline_config: dict[str, Any] = Field(default_factory=dict)
    proposal_config: dict[str, Any] = Field(default_factory=dict)


class ProposalResponse(BaseModel):
    proposal_id: UUID
    tenant_id: str
    failure_cluster_id: str | None = None
    status: str
    baseline_config: dict[str, Any]
    proposal_config: dict[str, Any]
    eval_run_id: UUID | None = None
    eval_result: dict[str, Any] | None = None
    approver: str | None = None
    approved_at: datetime | None = None
    canary_result: dict[str, Any] | None = None
    deployment_result: dict[str, Any] | None = None
    rolled_back_at: datetime | None = None
    rollback_reason: str | None = None
    created_at: datetime


class ProposalTransitionRequest(BaseModel):
    target_status: str
    notes: str | None = None
    eval_run_id: UUID | None = None
    eval_result: dict[str, Any] | None = None
    canary_result: dict[str, Any] | None = None
    deployment_result: dict[str, Any] | None = None
    rollback_reason: str | None = None


@router.get("/proposals", response_model=list[ProposalResponse])
async def list_proposals(
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
) -> list[ProposalResponse]:
    """List improvement proposals for the operator's tenant (or all if global)."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            conditions = (
                [] if operator.is_global else [ImprovementProposal.tenant_id == operator.tenant_id]
            )
            if status_filter:
                conditions.append(ImprovementProposal.status == status_filter)
            stmt = (
                select(ImprovementProposal)
                .where(*conditions)
                .order_by(ImprovementProposal.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            proposals = result.scalars().all()
    except Exception:
        proposals = []

    return [
        ProposalResponse(
            proposal_id=p.proposal_id,
            tenant_id=p.tenant_id,
            failure_cluster_id=p.failure_cluster_id,
            status=p.status,
            baseline_config=p.baseline_config,
            proposal_config=p.proposal_config,
            eval_run_id=p.eval_run_id,
            eval_result=p.eval_result,
            approver=p.approver,
            approved_at=p.approved_at,
            canary_result=p.canary_result,
            deployment_result=p.deployment_result,
            rolled_back_at=p.rolled_back_at,
            rollback_reason=p.rollback_reason,
            created_at=p.created_at,
        )
        for p in proposals
    ]


@router.post("/proposals", response_model=ProposalResponse, status_code=status.HTTP_201_CREATED)
async def create_proposal(
    proposal: ProposalSubmission,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
) -> ProposalResponse:
    """Create a new improvement proposal."""
    record = ImprovementProposal(
        tenant_id=operator.tenant_id,
        failure_cluster_id=proposal.failure_cluster_id,
        status="PROPOSED",
        baseline_config=proposal.baseline_config,
        proposal_config=proposal.proposal_config,
    )
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create proposal: {exc}",
        ) from exc

    return ProposalResponse(
        proposal_id=record.proposal_id,
        tenant_id=record.tenant_id,
        failure_cluster_id=record.failure_cluster_id,
        status=record.status,
        baseline_config=record.baseline_config,
        proposal_config=record.proposal_config,
        eval_run_id=record.eval_run_id,
        eval_result=record.eval_result,
        approver=record.approver,
        approved_at=record.approved_at,
        canary_result=record.canary_result,
        deployment_result=record.deployment_result,
        rolled_back_at=record.rolled_back_at,
        rollback_reason=record.rollback_reason,
        created_at=record.created_at,
    )


@router.post("/proposals/{proposal_id}/transition", response_model=ProposalResponse)
async def transition_proposal(  # noqa: PLR0912, PLR0915
    proposal_id: UUID,
    request: ProposalTransitionRequest,
    operator: Annotated[OperatorIdentity, Depends(_get_operator_identity)],
) -> ProposalResponse:
    """Transition proposal to next state."""
    if request.target_status not in PROPOSAL_STATES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target status: {request.target_status}",
        )

    proposal_not_found = False
    not_owner = False
    invalid_transition = False
    missing_guard = False
    transition_error_msg = ""
    current_status = ""

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            stmt = select(ImprovementProposal).where(ImprovementProposal.proposal_id == proposal_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()
            if record is None:
                proposal_not_found = True
            elif not operator.is_global and record.tenant_id != operator.tenant_id:
                not_owner = True
            else:
                current_status = record.status
                allowed = VALID_TRANSITIONS.get(current_status, set())
                if request.target_status not in allowed:
                    invalid_transition = True
                    transition_error_msg = (
                        f"Cannot transition from {current_status} to {request.target_status}"
                    )

            if not proposal_not_found and not not_owner and not invalid_transition:
                assert record is not None
                if request.target_status == "EVALUATED":
                    if not request.eval_run_id or not request.eval_result:
                        missing_guard = True
                        transition_error_msg = (
                            "Cannot transition to EVALUATED without both "
                            "eval_run_id and eval_result"
                        )
                    else:
                        record.eval_run_id = request.eval_run_id
                        record.eval_result = request.eval_result
                        record.status = request.target_status
                elif request.target_status == "APPROVED":
                    if not request.eval_result:
                        missing_guard = True
                        transition_error_msg = "Cannot transition to APPROVED without eval_result"
                    else:
                        record.approver = operator.operator_id
                        record.approved_at = datetime.now(UTC)
                        record.status = request.target_status
                elif request.target_status == "CANARY":
                    if current_status != "APPROVED":
                        missing_guard = True
                        transition_error_msg = "Canary requires prior approval"
                    else:
                        record.status = request.target_status
                elif request.target_status == "PROMOTED":
                    if not request.deployment_result:
                        missing_guard = True
                        transition_error_msg = (
                            "Cannot transition to PROMOTED without deployment_result"
                        )
                    else:
                        record.deployment_result = request.deployment_result
                        record.status = request.target_status
                elif request.target_status == "ROLLED_BACK":
                    record.rolled_back_at = datetime.now(UTC)
                    record.rollback_reason = request.rollback_reason
                    record.status = request.target_status
                else:
                    record.status = request.target_status

                await session.commit()
                await session.refresh(record)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transition proposal: {exc}",
        ) from exc

    if proposal_not_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Proposal {proposal_id} not found",
        )
    if not_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify proposals from other tenants",
        )
    if invalid_transition:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=transition_error_msg,
        )
    if missing_guard:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=transition_error_msg,
        )

    assert record is not None
    return ProposalResponse(
        proposal_id=record.proposal_id,
        tenant_id=record.tenant_id,
        failure_cluster_id=record.failure_cluster_id,
        status=record.status,
        baseline_config=record.baseline_config,
        proposal_config=record.proposal_config,
        eval_run_id=record.eval_run_id,
        eval_result=record.eval_result,
        approver=record.approver,
        approved_at=record.approved_at,
        canary_result=record.canary_result,
        deployment_result=record.deployment_result,
        rolled_back_at=record.rolled_back_at,
        rollback_reason=record.rollback_reason,
        created_at=record.created_at,
    )
