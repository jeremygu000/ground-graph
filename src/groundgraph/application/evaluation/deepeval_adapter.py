"""DeepEval metrics adapter for GroundGraph.

Provides GroundGraph-specific evaluation metrics backed by DeepEval.
Each metric conforms to the Evaluator interface used by the evaluation
runner in `evals/runners/`.

Metrics implemented:
  - faithfulness: measures whether the LLM response is grounded in the provided evidence
  - answer_relevancy: measures how relevant the answer is to the question

Usage:
    from groundgraph.application.evaluation.deepeval_adapter import (
        GroundGraphFaithfulnessMetric,
        GroundGraphAnswerRelevancyMetric,
    )

Requires `deepeval` to be installed (optional dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from groundgraph.domain.retrieval import Evidence, QueryResponse

try:
    from deepeval.metrics import AnswerRelevancyMetric as _DeepEvalAnswerRelevancy
    from deepeval.metrics import FaithfulnessMetric as _DeepEvalFaithfulness
    from deepeval.test_case import LLMTestCase

    _deepeval_installed = True
except ImportError:
    _deepeval_installed = False
    _DeepEvalFaithfulness = None  # type: ignore[assignment, misc]
    _DeepEvalAnswerRelevancy = None  # type: ignore[assignment, misc]
    LLMTestCase = None  # type: ignore[assignment, misc]

_DEEPEVAL_AVAILABLE: bool = _deepeval_installed


@dataclass
class EvaluationResult:
    """Result of a single metric evaluation."""

    metric: str
    score: float
    reason: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def _build_test_case(
    question: str,
    answer: str | None,
    evidence: list[Evidence],
) -> Any:
    """Build a DeepEval LLMTestCase from GroundGraph types."""
    context = "\n".join(f"[{i + 1}] {e.content[:1000]}" for i, e in enumerate(evidence))
    retrieval_context: list[str] = [e.content[:1000] for e in evidence]
    return cast(Any, LLMTestCase)(
        input=question,
        actual_output=answer or "",
        context=[context],
        retrieval_context=cast(Any, retrieval_context),
    )


class GroundGraphFaithfulnessMetric:
    """Faithfulness metric using DeepEval.

    Measures whether the LLM response is factually grounded in the
    provided evidence chunks.
    """

    name = "faithfulness"

    def __init__(self, threshold: float = 0.5) -> None:
        if not _DEEPEVAL_AVAILABLE:
            raise RuntimeError("deepeval is not installed. Install with: uv add deepeval")
        self._metric = cast(Any, _DeepEvalFaithfulness)(threshold=threshold)

    async def evaluate(
        self,
        response: QueryResponse,
        evidence: list[Evidence],
    ) -> EvaluationResult:
        """Evaluate faithfulness of response given evidence."""
        if not evidence:
            return EvaluationResult(
                metric=self.name,
                score=0.0,
                reason="No evidence provided",
            )
        if not _DEEPEVAL_AVAILABLE:  # pragma: no cover
            return EvaluationResult(
                metric=self.name,
                score=0.0,
                error="deepeval not available",
            )

        try:
            test_case = _build_test_case(
                question="",
                answer=response.answer,
                evidence=evidence,
            )
            score = self._metric.measure(test_case, _show_indicator=False)
            return EvaluationResult(
                metric=self.name,
                score=score,
            )
        except Exception as exc:  # pragma: no cover
            return EvaluationResult(
                metric=self.name,
                score=0.0,
                error=str(exc),
            )


class GroundGraphAnswerRelevancyMetric:
    """Answer relevancy metric using DeepEval.

    Measures how relevant the generated answer is to the original question.
    """

    name = "answer_relevancy"

    def __init__(self, threshold: float = 0.3) -> None:
        if not _DEEPEVAL_AVAILABLE:
            raise RuntimeError("deepeval is not installed. Install with: uv add deepeval")
        self._metric = cast(Any, _DeepEvalAnswerRelevancy)(threshold=threshold)

    async def evaluate(
        self,
        response: QueryResponse,
        evidence: list[Evidence],
    ) -> EvaluationResult:
        """Evaluate answer relevancy."""
        if not _DEEPEVAL_AVAILABLE:  # pragma: no cover
            return EvaluationResult(
                metric=self.name,
                score=0.0,
                error="deepeval not available",
            )

        try:
            test_case = _build_test_case(
                question="",
                answer=response.answer,
                evidence=evidence,
            )
            score = self._metric.measure(test_case, _show_indicator=False)
            return EvaluationResult(
                metric=self.name,
                score=score,
            )
        except Exception as exc:  # pragma: no cover
            return EvaluationResult(
                metric=self.name,
                score=0.0,
                error=str(exc),
            )


class GroundGraphCitationMetric:
    """Citation accuracy metric (custom, no DeepEval needed).

    Measures whether all cited evidence IDs correspond to actual retrieved evidence.
    """

    name = "citation_accuracy"

    def __init__(self, threshold: float = 1.0) -> None:
        self._threshold = threshold

    async def evaluate(
        self,
        response: QueryResponse,
        evidence: list[Evidence],
    ) -> EvaluationResult:
        """Evaluate citation accuracy.

        A citation is accurate if every cited evidence_id appears in the evidence list.
        """
        if not evidence:
            reason = "No evidence available for validation"
            score = 0.5
        elif not response.citations:
            reason = "No citations in response"
            score = 1.0 if not evidence else 0.5
        else:
            evidence_ids = {e.evidence_id for e in evidence}
            cited_ids = {c.evidence_id for c in response.citations}
            accurate = cited_ids.issubset(evidence_ids)
            if accurate:
                reason = f"All {len(cited_ids)} cited IDs found in evidence"
                score = 1.0
            else:
                missing = cited_ids - evidence_ids
                reason = f"{len(missing)} cited IDs not in evidence"
                score = 0.0

        return EvaluationResult(
            metric=self.name,
            score=score,
            reason=reason,
        )


class GroundGraphNoHallucinationMetric:
    """Hallucination detection metric (custom heuristic).

    Flags responses that make claims not supported by any evidence.
    """

    name = "no_hallucination"

    def __init__(self) -> None:
        pass

    async def evaluate(
        self,
        response: QueryResponse,
        evidence: list[Evidence],
    ) -> EvaluationResult:
        """Evaluate for hallucination."""
        if not evidence:
            reason = "No evidence available for validation"
            score = 0.5
        elif not response.answer:
            reason = "Empty response"
            score = 0.0
        else:
            unsupported = [c for c in response.claims if c.support_status == "unsupported"]
            if unsupported:
                reason = f"{len(unsupported)} unsupported claims found"
                score = 0.0
            else:
                reason = "All claims supported"
                score = 1.0

        return EvaluationResult(
            metric=self.name,
            score=score,
            reason=reason,
        )


@dataclass
class EvaluationCase:
    """A single evaluation case from a JSONL dataset."""

    id: str
    question: str
    tenant_id: str
    principal: str
    expected_status: str | None = None
    required_claim_text: str | None = None
    forbidden_claim_text: list[str] | None = None
    expected_entities: list[str] | None = None
    metadata: dict[str, Any] | None = None


def load_jsonl_dataset(path: str) -> list[EvaluationCase]:
    """Load evaluation cases from a JSONL file.

    Each line must be a JSON object with the following required fields:
      - id: case identifier
      - question: the query to evaluate
      - tenant_id: tenant for the query
      - principal: principal for the query

    Optional fields:
      - expected_status: expected response status
      - required_claim_text: text that must appear in the answer
      - forbidden_claim_text: text that must NOT appear in the answer
      - expected_entities: entities expected to be in the answer
      - metadata: additional case metadata

    Args:
        path: Path to the JSONL file

    Returns:
        List of EvaluationCase objects

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if a line is not valid JSON or missing required fields
    """
    cases: list[EvaluationCase] = []
    with open(path) as f:
        for line_no, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            try:
                case = EvaluationCase(
                    id=data["id"],
                    question=data["question"],
                    tenant_id=data["tenant_id"],
                    principal=data["principal"],
                    expected_status=data.get("expected_status"),
                    required_claim_text=data.get("required_claim_text"),
                    forbidden_claim_text=data.get("forbidden_claim_text"),
                    expected_entities=data.get("expected_entities"),
                    metadata=data.get("metadata"),
                )
                cases.append(case)
            except KeyError as exc:
                raise ValueError(f"Line {line_no}: missing required field {exc}") from exc
    return cases
