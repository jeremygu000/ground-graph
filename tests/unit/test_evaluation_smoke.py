"""Tests for the evaluation smoke module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from groundgraph.application.evaluation.smoke import (
    _check_dataset,
    _check_domain_models,
    _check_ports_importable,
    _check_workflow,
    _run,
)


class TestCheckDataset:
    """Tests for the dataset validation function."""

    def test_missing_cases_key(self, tmp_path: Path) -> None:
        (tmp_path / "test.json").write_text(json.dumps({"not_cases": []}))
        ok, errors = _check_dataset(tmp_path / "test.json")
        assert not ok
        assert any("missing 'cases'" in e for e in errors)

    def test_cases_not_a_list(self, tmp_path: Path) -> None:
        (tmp_path / "test.json").write_text(json.dumps({"cases": {}}))
        ok, errors = _check_dataset(tmp_path / "test.json")
        assert not ok
        assert any("no cases" in e for e in errors)

    def test_empty_cases(self, tmp_path: Path) -> None:
        (tmp_path / "test.json").write_text(json.dumps({"cases": []}))
        ok, errors = _check_dataset(tmp_path / "test.json")
        assert not ok
        assert any("no cases" in e for e in errors)

    def test_valid_m4_dataset(self, tmp_path: Path) -> None:
        data = {
            "cases": [
                {
                    "id": "test-001",
                    "type": "factual",
                    "question": "What is Postgres?",
                    "tenantId": "t1",
                    "principalId": "p1",
                    "expectedStatus": "answered",
                }
            ]
        }
        (tmp_path / "m4-vector-baseline.json").write_text(json.dumps(data))
        ok, errors = _check_dataset(tmp_path / "m4-vector-baseline.json")
        assert ok
        assert errors == []

    def test_valid_m6_dataset(self, tmp_path: Path) -> None:
        data = {
            "cases": [
                {
                    "id": "mh-001",
                    "type": "multi-hop",
                    "question": "What is the dependency chain?",
                    "tenant_id": "t1",
                    "principal": "p1",
                    "seed_entity": "A",
                    "expected_predicate": "DEPENDS_ON",
                }
            ]
        }
        (tmp_path / "m6-hybrid-graph-retrieval-v1.json").write_text(json.dumps(data))
        ok, errors = _check_dataset(tmp_path / "m6-hybrid-graph-retrieval-v1.json")
        assert ok
        assert errors == []

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json")
        ok, errors = _check_dataset(tmp_path / "bad.json")
        assert not ok
        assert any("Failed to load" in e for e in errors)


class TestCheckDomainModels:
    """Tests for domain model instantiation."""

    def test_domain_models_instantiate(self) -> None:
        ok, errors = _check_domain_models()
        assert ok
        assert errors == []


class TestRun:
    """Tests for the main smoke test runner."""

    def test_run_all_pass(self) -> None:
        with patch(
            "groundgraph.application.evaluation.smoke.DATASET_FILES",
            [],
        ):
            ok = _run()
            assert ok is True


class TestCheckWorkflow:
    """Tests for workflow import check."""

    def test_workflow_importable(self) -> None:
        ok, errors = _check_workflow()
        assert ok
        assert errors == []


class TestCheckPorts:
    """Tests for ports import check."""

    def test_ports_importable(self) -> None:
        ok, errors = _check_ports_importable()
        assert ok
        assert errors == []
