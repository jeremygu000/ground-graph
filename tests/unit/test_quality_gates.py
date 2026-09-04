"""Quality-gate wiring tests.

These tests assert that the three quality gates (format, lint, type-check)
plus tests are properly wired across the repository:

- ``pyproject.toml`` declares ruff (format+lint), pyright, and mypy
- ``Makefile`` exposes ``format``, ``lint-check``, ``typecheck``, ``test``,
  and ``check`` (which runs all four)
- ``.github/workflows/ci.yml`` runs all four gates
- ``.pre-commit-config.yaml`` wires the same gates locally

These tests catch accidental removal of a gate (e.g. someone deleting
the typecheck step because it slows them down).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
MAKEFILE = ROOT / "Makefile"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT = ROOT / ".pre-commit-config.yaml"


def _read(path: Path) -> str:
    assert path.exists(), f"Required file missing: {path}"
    return path.read_text(encoding="utf-8")


def test_pyproject_declares_ruff() -> None:
    text = _read(PYPROJECT)
    assert '"ruff>=0.7"' in text or "ruff>=0.7" in text, (
        "pyproject.toml must pin ruff as a dev dependency"
    )
    assert "[tool.ruff]" in text
    assert "[tool.ruff.lint]" in text
    assert "[tool.ruff.format]" in text


def test_pyproject_declares_pyright() -> None:
    text = _read(PYPROJECT)
    assert "pyright>=1.1.380" in text, "pyproject.toml must pin pyright"
    assert "[tool.pyright]" in text


def test_pyproject_declares_mypy() -> None:
    text = _read(PYPROJECT)
    assert "mypy>=1.13.0" in text, "pyproject.toml must pin mypy"
    assert "[tool.mypy]" in text
    assert "strict = true" in text, "mypy must run strict on domain/application"


def test_pyright_targets_domain_and_application_strictly() -> None:
    text = _read(PYPROJECT)
    match = re.search(r"\[tool\.pyright\](.*?)(?=\n\[)", text, re.DOTALL)
    assert match, "[tool.pyright] block not found"
    block = match.group(1)
    assert "src/groundgraph/domain" in block
    assert "src/groundgraph/application" in block


def test_coverage_gate_is_enforced() -> None:
    text = _read(PYPROJECT)
    match = re.search(r"\[tool\.coverage\.report\](.*?)(?=\n\[)", text, re.DOTALL)
    assert match, "[tool.coverage.report] block not found"
    block = match.group(1)
    assert "fail_under" in block, "coverage must enforce a fail_under threshold"
    # Extract the numeric value
    val_match = re.search(r"fail_under\s*=\s*(\d+)", block)
    assert val_match
    val = int(val_match.group(1))
    assert 70 <= val <= 95, (
        f"coverage fail_under {val} is outside the expected 70-95 band; "
        "plan.md section 14.3 sets the initial gate at 85%"
    )


def test_makefile_check_does_not_modify_workspace() -> None:
    """`make check` must use ruff format --check, not the writing form.

    `make format` may rewrite files; `make check` must be read-only so it
    is safe to run in CI and as a pre-push gate.
    """
    text = _read(MAKEFILE)
    match = re.search(r"^check:\s*([^\n]+)", text, re.MULTILINE)
    assert match
    deps = match.group(1)
    # Must include the read-only formatter target.
    assert "format-check" in deps, (
        "`make check` must depend on format-check, not on `format` (which writes)"
    )
    # The bare `format` target must not be a direct dependency of `check`.
    tokens = re.split(r"[\s]+", deps.strip())
    assert "format" not in tokens, (
        f"`make check` must not depend on the writing `format` target; got tokens {tokens}"
    )


def test_settings_module_renamed() -> None:
    """No leftover references to the old package namespace in runtime code."""
    result = subprocess.run(
        [
            "grep",
            "-rln",
            r"\bgraphrag\.",
            "src",
            "apps",
            "--include=*.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # grep returns 1 when no match — that's success here.
    if result.returncode == 0 and result.stdout.strip():
        pytest.fail("Found references to the old package namespace:\n" + result.stdout)


def test_pyproject_hatch_package_is_groundgraph() -> None:
    text = _read(PYPROJECT)
    assert 'packages = ["src/groundgraph"]' in text, (
        "hatch build target must point at src/groundgraph"
    )


def test_precommit_dependency_is_declared() -> None:
    text = _read(PYPROJECT)
    assert "pre-commit" in text, "pre-commit must be in dev dependencies"


def test_makefile_exposes_required_targets() -> None:
    text = _read(MAKEFILE)
    required = {
        "format": r"^format:",
        "lint-check": r"^lint-check:",
        "typecheck": r"^typecheck:",
        "test": r"^test:",
        "check": r"^check:",
    }
    for name, pattern in required.items():
        assert re.search(pattern, text, re.MULTILINE), f"Makefile must define a {name!r} target"


def test_makefile_check_runs_all_gates() -> None:
    text = _read(MAKEFILE)
    match = re.search(r"^check:\s*([^\n]+)", text, re.MULTILINE)
    assert match, "check target not found"
    deps = match.group(1)
    # `test-cov` is required (not bare `test`) so the coverage fail_under
    # gate actually runs as part of `make check` and CI.
    for gate in ("format-check", "lint-check", "typecheck", "test-cov"):
        assert gate in deps, f"make check must depend on {gate!r}; got {deps!r}"


def test_makefile_test_cov_runs_coverage_gate() -> None:
    """test-cov must pass --cov-fail-under=85 so the gate is real."""
    text = _read(MAKEFILE)
    # Makefiles indent recipe lines with TAB, not spaces.
    match = re.search(r"^test-cov:[^\n]*\n((?:\t[^\n]*\n)+)", text, re.MULTILINE)
    assert match, "test-cov target not found"
    body = match.group(1)
    assert "--cov" in body, "test-cov must run pytest with --cov"
    assert "cov-fail-under" in body, "test-cov must pass --cov-fail-under=85 to make the gate real"


def test_ci_workflow_runs_coverage_gate() -> None:
    """GitHub Actions must run the coverage command, not bare pytest."""
    parsed = yaml.safe_load(_read(CI_YML))
    runs: list[str] = []
    for job in parsed.get("jobs", {}).values():
        for step in job.get("steps", []) or []:
            run = step.get("run")
            if isinstance(run, str):
                runs.append(run)
    joined = "\n".join(runs)
    assert "cov-fail-under" in joined, (
        "CI must run pytest with --cov-fail-under so the gate is enforced"
    )


def test_makefile_typecheck_runs_pyright_and_mypy() -> None:
    text = _read(MAKEFILE)
    match = re.search(r"^typecheck:\s*([^\n]+)", text, re.MULTILINE)
    assert match, "typecheck target not found"
    body = match.group(1)
    assert "pyright" in body, "typecheck must run pyright"
    assert "mypy" in body, "typecheck must run mypy"


def test_ci_runs_all_quality_gates() -> None:
    parsed = yaml.safe_load(_read(CI_YML))
    # Concatenate all `run:` values into one searchable string
    runs: list[str] = []
    for job in parsed.get("jobs", {}).values():
        for step in job.get("steps", []) or []:
            run = step.get("run")
            if isinstance(run, str):
                runs.append(run)
    joined = "\n".join(runs)
    assert "ruff format --check" in joined, "CI must run ruff format --check"
    assert "ruff check" in joined, "CI must run ruff check (lint)"
    assert "pyright" in joined, "CI must run pyright"
    assert "mypy" in joined, "CI must run mypy"
    assert "pytest" in joined, "CI must run pytest"


def test_precommit_runs_quality_gates() -> None:
    text = _read(PRECOMMIT)
    assert "ruff format" in text
    assert "ruff check" in text or "ruff lint" in text
    assert "pyright" in text
    # Note: pre-commit is intentionally lighter than CI (no mypy by default
    # to keep the hook fast), but the Makefile + CI still cover it.


@pytest.mark.parametrize(
    "cmd",
    [
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pyright", "--version"],
        ["uv", "run", "mypy", "--version"],
    ],
)
def test_quality_tools_are_installed(cmd: list[str]) -> None:
    """Smoke check: the three tools are actually installed in the uv env."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"{' '.join(cmd)} failed: {proc.stdout}\n{proc.stderr}"
