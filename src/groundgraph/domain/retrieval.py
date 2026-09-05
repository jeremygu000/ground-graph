"""Retrieval / answer domain contracts (plan.md §4.3).

Pure Pydantic v2 — no infrastructure imports.

These are the contracts that flow between the retrieval planner,
evidence reranker, answer generator, and the API. They are
deliberately not coupled to any specific LLM provider or
embedding model — the ``retrieval_method`` and ``support_status``
literals are the extension points.
"""

# Note: do NOT use ``from __future__ import annotations`` here.
# Pyright cannot resolve ``Literal[...]`` field types when annotations
# are deferred to strings, and Pydantic v2 evaluates them at class
# build time anyway — keeping annotations as real objects gives both
# pyright and Pydantic the same view.

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundgraph.domain.defaults import (
    empty_answer_claim_list,
    empty_citation_list,
    empty_resolved_entity_list,
    empty_str_list,
    empty_uuid_list,
)


class ResolvedEntity(BaseModel):
    """An entity that the planner has already resolved (post-entity-resolution).

    Used as a query hint: "the user is talking about this entity".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: UUID
    canonical_name: str
    entity_type: str


class RetrievalPlan(BaseModel):
    """The query plan the planner emits before any retrieval happens.

    The plan is itself an audit artifact: every retrieval_run row
    references the plan that produced it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["vector", "graph", "hybrid"]
    question_type: Literal[
        "fact",
        "relationship",
        "multi_hop",
        "temporal",
        "comparison",
        "impact",
        "summary",
        "unknown",
    ]
    query_texts: list[str] = Field(default_factory=empty_str_list)
    entities: list[ResolvedEntity] = Field(default_factory=empty_resolved_entity_list)
    predicates: list[str] = Field(default_factory=empty_str_list)
    max_graph_depth: int = Field(default=2, ge=0, le=3)
    vector_top_k: int = Field(default=10, ge=1, le=50)
    final_evidence_limit: int = Field(default=10, ge=1, le=30)
    valid_at: datetime | None = None
    reason_codes: list[str] = Field(default_factory=empty_str_list)


class Citation(BaseModel):
    """A pointer from a claim back to a single evidence unit.

    The citation is the bridge between prose the user reads and
    the source-of-truth evidence that the model saw. It must
    include the principal list because a citation shown to a
    user who is not allowed to read the source is itself an
    information leak.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_id: UUID
    claim_id: UUID
    evidence_id: UUID
    locator: str
    allowed_principals: list[str] = Field(default_factory=empty_str_list)


class Evidence(BaseModel):
    """One piece of evidence backing (or refuting) an answer claim.

    ``retrieval_method`` is the producer: a vector search, a keyword
    search, a graph walk, or a structured-record lookup. The
    downstream claim validator uses this to weight how much the
    evidence should count.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: UUID
    source_id: UUID
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    structured_record_id: str | None = None
    content: str
    retrieval_method: Literal["vector", "keyword", "graph", "structured"]
    vector_score: float | None = None
    rerank_score: float | None = None
    graph_path_fact_ids: list[UUID] = Field(default_factory=empty_uuid_list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    allowed_principals: list[str] = Field(default_factory=empty_str_list)

    @model_validator(mode="after")
    def _validate_temporal_bounds(self) -> "Evidence":
        if self.valid_from is not None and self.valid_from.tzinfo is None:
            raise ValueError("valid_from must be timezone-aware")
        if self.valid_to is not None and self.valid_to.tzinfo is None:
            raise ValueError("valid_to must be timezone-aware")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            raise ValueError("valid_to must be greater than valid_from")
        return self


class AnswerClaim(BaseModel):
    """One atomic claim in the answer.

    ``support_status`` is "supported" only when there is at least
    one citation that fully covers the claim text and the cited
    evidence was produced by a method that can ground the claim
    type. A claim may be ``unsupported``; the user must be able
    to see that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: UUID
    text: str
    factual: bool
    evidence_ids: list[UUID] = Field(default_factory=empty_uuid_list)
    support_status: Literal["supported", "partially_supported", "unsupported"]

    @model_validator(mode="after")
    def _validate_evidence_for_support_status(self) -> "AnswerClaim":
        has_evidence = len(self.evidence_ids) > 0
        if self.support_status in ("supported", "partially_supported") and not has_evidence:
            raise ValueError(
                f"support_status '{self.support_status}' requires at least one evidence_id"
            )
        if self.support_status == "unsupported" and has_evidence:
            raise ValueError("support_status 'unsupported' requires no evidence_ids")
        return self


class QueryResponse(BaseModel):
    """The full response the API returns to the user.

    ``answer`` may be None if the workflow chose
    ``status="clarification_required"`` or
    ``"insufficient_evidence"``. ``warnings`` is for soft
    conditions (e.g. "3 of 12 chunks were filtered for ACL");
    errors are a separate channel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_run_id: UUID
    answer: str | None = None
    status: Literal[
        "answered",
        "clarification_required",
        "insufficient_evidence",
        "failed",
    ]
    claims: list[AnswerClaim] = Field(default_factory=empty_answer_claim_list)
    citations: list[Citation] = Field(default_factory=empty_citation_list)
    confidence_band: Literal["high", "medium", "low"]
    warnings: list[str] = Field(default_factory=empty_str_list)

    @model_validator(mode="after")
    def _validate_answer_status_consistency(self) -> "QueryResponse":
        if self.status == "answered":
            if self.answer is None:
                raise ValueError("answer must be non-None when status is 'answered'")
            if self.answer == "":
                raise ValueError("answer must be non-empty when status is 'answered'")
        elif self.answer is not None:
            raise ValueError("answer must be None when status is not 'answered'")
        return self
