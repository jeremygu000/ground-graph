"""Domain contracts smoke tests.

These tests ensure the package skeleton is importable and that domain
modules do not pull in framework libraries.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "graphrag.domain",
        "graphrag.application",
        "graphrag.application.settings",
        "graphrag.application.errors",
        "graphrag.application.result",
    ],
)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_domain_package_does_not_import_frameworks() -> None:
    """The domain package must not import any framework library.

    Runs in a fresh subprocess so we can observe which modules are
    pulled in as a side effect of ``import graphrag.domain`` without
    inheriting modules already loaded by other tests in this session.
    """
    snippet = textwrap.dedent(
        """
        import sys
        import graphrag.domain  # noqa: F401
        forbidden = {
            "fastapi", "uvicorn", "langgraph", "langchain", "openai",
            "neo4j", "sqlalchemy", "alembic", "asyncpg", "psycopg",
            "pgvector", "phoenix", "arize", "boto3", "minio",
            "opentelemetry",
        }
        offenders = sorted(
            name for name in sys.modules
            if name.split(".")[0] in forbidden
        )
        if offenders:
            print("OFFENDERS=" + ",".join(offenders))
            sys.exit(1)
        sys.exit(0)
        """
    ).strip()

    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        offenders = ""
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            if line.startswith("OFFENDERS="):
                offenders = line.split("=", 1)[1]
                break
        pytest.fail(f"Domain package must not import framework libraries; offenders: {offenders}")
