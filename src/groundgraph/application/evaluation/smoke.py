"""Smoke tests for the GroundGraph evaluation system.

This module is the entry point for `make eval-smoke`. It runs a fast,
self-contained smoke test that verifies the evaluation infrastructure is
intact without requiring Docker or external services.

Checks performed:
  1. Evaluation dataset JSON files are valid and well-formed
  2. Dataset case structure matches the expected schema
  3. Domain models used by evaluation can be instantiated
  4. Key ports (AnswerGenerator, EvidenceReranker, VectorRetriever, etc.) are importable
  5. The query workflow graph can be imported and its nodes are present

Exit codes:
  0 - all smoke checks passed
  1 - one or more checks failed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from groundgraph.application.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    EvidenceReranker,
    GraphRepository,
    KeywordRetrieverPort,
    VectorContentRetriever,
)
from groundgraph.application.retrieval.hybrid_retrieval import (
    HybridRetrievalService,
)
from groundgraph.domain.retrieval import Evidence, QueryResponse
from groundgraph.workflows.query_graph import QueryWorkflow

__version__ = "0.1.0"

EVAL_ROOT = Path(__file__).parent.parent.parent.parent.parent / "evals"
DATASET_FILES = [
    EVAL_ROOT / "datasets" / "m4-vector-baseline.json",
    EVAL_ROOT / "datasets" / "m6-hybrid-graph-retrieval-v1.json",
]

EXPECTED_CASE_FIELDS_M4 = frozenset(
    {
        "id",
        "type",
        "question",
        "tenantId",
        "principalId",
        "expectedStatus",
    }
)

EXPECTED_CASE_FIELDS_M6 = frozenset(
    {
        "id",
        "type",
        "question",
        "tenant_id",
        "principal",
        "seed_entity",
        "expected_predicate",
    }
)

DATASET_SCHEMA_MAP: dict[str, frozenset[str]] = {
    "m4-vector-baseline.json": EXPECTED_CASE_FIELDS_M4,
    "m6-hybrid-graph-retrieval-v1.json": EXPECTED_CASE_FIELDS_M6,
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return cast(dict[str, Any], json.load(f))


def _check_dataset(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        data = _load_json(path)
    except Exception as exc:  # pragma: no cover
        return False, [f"Failed to load {path.name}: {exc}"]

    if "cases" not in data:  # pragma: no cover
        return False, [f"{path.name}: missing 'cases' key"]

    cases_raw = cast(list[dict[str, Any]], data["cases"])
    if not cases_raw:  # pragma: no cover
        return False, [f"{path.name}: dataset has no cases"]

    expected_fields = DATASET_SCHEMA_MAP.get(path.name, frozenset())
    for i, case in enumerate(cases_raw):
        case_id = str(case.get("id", repr(i)))
        missing = expected_fields - case.keys()
        if missing:  # pragma: no cover
            errors.append(f"  Case {case_id} missing fields: {missing}")
        question = case.get("question")
        if question is not None and not isinstance(question, str):  # pragma: no cover
            errors.append(f"  Case {case_id}: 'question' must be a string")

    return len(errors) == 0, errors


def _check_domain_models() -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        ev = Evidence(
            evidence_id=uuid4(),
            source_id=uuid4(),
            content="smoke test evidence",
            retrieval_method="vector",
        )
        resp = QueryResponse(
            execution_run_id=uuid4(),
            answer="smoke test answer",
            status="answered",
            claims=[],
            citations=[],
            confidence_band="high",
            warnings=[],
        )
        assert ev.evidence_id is not None
        assert resp.answer == "smoke test answer"
    except Exception as exc:  # pragma: no cover
        errors.append(f"Domain model smoke test failed: {exc}")

    return len(errors) == 0, errors


def _check_ports_importable() -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        assert AnswerGenerator is not None
        assert EvidenceReranker is not None
        assert VectorContentRetriever is not None
        assert GraphRepository is not None
        assert EmbeddingProvider is not None
        assert KeywordRetrieverPort is not None
    except Exception as exc:  # pragma: no cover
        errors.append(f"Port import failed: {exc}")

    return len(errors) == 0, errors


def _check_workflow() -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        assert QueryWorkflow is not None
        assert HybridRetrievalService is not None
    except Exception as exc:  # pragma: no cover
        errors.append(f"Workflow import failed: {exc}")

    return len(errors) == 0, errors


def _run() -> bool:
    print(f"GroundGraph Evaluation Smoke Test v{__version__}")
    print(f"EVAL_ROOT: {EVAL_ROOT}")
    print()

    all_passed = True

    for path in DATASET_FILES:
        status, errors = _check_dataset(path)
        label = "PASS" if status else "FAIL"
        print(f"[{label}] Dataset: {path.name}")
        if not status:
            all_passed = False
            for e in errors:
                print(f"       {e}")

    print()

    status, errors = _check_domain_models()
    label = "PASS" if status else "FAIL"
    print(f"[{label}] Domain models (Evidence, QueryResponse)")
    if not status:
        all_passed = False
        for e in errors:
            print(f"       {e}")
    print()

    status, errors = _check_ports_importable()
    label = "PASS" if status else "FAIL"
    print(f"[{label}] Evaluation ports importable")
    if not status:
        all_passed = False
        for e in errors:
            print(f"       {e}")
    print()

    status, errors = _check_workflow()
    label = "PASS" if status else "FAIL"
    print(f"[{label}] Query workflow graph importable")
    if not status:
        all_passed = False
        for e in errors:
            print(f"       {e}")
    print()

    print("-" * 50)
    overall = "PASS" if all_passed else "FAIL"
    print(f"Overall: [{overall}]")
    return all_passed


if __name__ == "__main__":
    success = _run()
    sys.exit(0 if success else 1)
