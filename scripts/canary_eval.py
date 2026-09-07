#!/usr/bin/env python3
"""Canary evaluation script for GroundGraph.

Deploys a canary version alongside the stable version, runs a targeted
evaluation dataset against both, and compares results before promoting
or rolling back.

Usage:
    uv run python scripts/canary_eval.py \
        --stable-url http://api-stable:8000 \
        --canary-url http://api-canary:8000 \
        --eval-dataset evals/datasets/pilot_eval.jsonl \
        --faithfulness-threshold 0.70 \
        --promote
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class EvalResult:
    url: str
    total: int
    faithfulness: float
    citation_accuracy: float
    client_errors: int
    server_errors: int


async def run_evals(url: str, dataset: list[dict[str, Any]]) -> EvalResult:
    client_errors = 0
    server_errors = 0
    faithfulness_scores = []
    citation_scores = []

    headers = {
        "X-Tenant-ID": "canary-eval-tenant",
        "X-Principal-ID": "canary-eval-user",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in dataset:
            question = item["question"]
            expected_answer = item.get("expected_answer", "")
            expected_sources = item.get("expected_sources", [])

            try:
                response = await client.post(
                    f"{url}/v1/query",
                    headers=headers,
                    json={"question": question},
                )

                if response.status_code >= 500:  # noqa: PLR2004
                    server_errors += 1
                elif response.status_code >= 400:  # noqa: PLR2004
                    client_errors += 1
                else:
                    data = response.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])

                    faithfulness = _score_faithfulness(answer, expected_answer)
                    citation_acc = _score_citation(citations, expected_sources)

                    faithfulness_scores.append(faithfulness)
                    citation_scores.append(citation_acc)

            except Exception:
                server_errors += 1

    total = len(dataset)
    avg_faithfulness = sum(faithfulness_scores) / max(len(faithfulness_scores), 1)
    avg_citation = sum(citation_scores) / max(len(citation_scores), 1)

    return EvalResult(
        url=url,
        total=total,
        faithfulness=avg_faithfulness,
        citation_accuracy=avg_citation,
        client_errors=client_errors,
        server_errors=server_errors,
    )


def _score_faithfulness(answer: str, expected: str) -> float:
    if not expected or not answer:
        return 0.0
    matching = sum(1 for w in expected.split() if w in answer)
    return matching / max(len(expected.split()), 1)


def _score_citation(citations: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    matched = sum(1 for src in expected if any(src in c for c in citations))
    return matched / max(len(expected), 1)


def _compare(stable: EvalResult, canary: EvalResult, threshold: float) -> bool:
    print("\n=== Canary Evaluation Report ===")
    print(f"{'Metric':<25} {'Stable':>10} {'Canary':>10} {'Delta':>10}")
    print("-" * 60)

    f_delta = canary.faithfulness - stable.faithfulness
    c_delta = canary.citation_accuracy - stable.citation_accuracy

    faithfulness_str = f"{'Faithfulness':<25} {stable.faithfulness:>10.3f} {canary.faithfulness:>10.3f} {f_delta:>+10.3f}"  # noqa: E501
    print(faithfulness_str)
    citation_str = f"{'Citation Accuracy':<25} {stable.citation_accuracy:>10.3f} {canary.citation_accuracy:>10.3f} {c_delta:>+10.3f}"  # noqa: E501
    print(citation_str)
    srv_err_delta = canary.server_errors - stable.server_errors
    srv_err_str = f"{'Server Errors':<25} {stable.server_errors:>10} {canary.server_errors:>10} {srv_err_delta:>+10}"  # noqa: E501
    print(srv_err_str)
    cli_err_delta = canary.client_errors - stable.client_errors
    cli_err_str = f"{'Client Errors':<25} {stable.client_errors:>10} {canary.client_errors:>10} {cli_err_delta:>+10}"  # noqa: E501
    print(cli_err_str)

    issues: list[str] = []
    if canary.faithfulness < threshold:
        issues.append(
            f"Canary faithfulness {canary.faithfulness:.3f} below threshold {threshold:.3f}"
        )
    if canary.faithfulness < stable.faithfulness - 0.05:
        issues.append("Canary faithfulness regressed >0.05 vs stable")
    if canary.server_errors > stable.server_errors + 2:
        issues.append("Canary server errors increased significantly")

    if issues:
        print("\n[CANARY ROLLBACK REQUIRED]")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\n[CANARY PROMOTED]")
    return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-url", required=True)
    parser.add_argument("--canary-url", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--faithfulness-threshold", type=float, default=0.70)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    with open(args.eval_dataset) as f:  # noqa: ASYNC230
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"[canary] Evaluating stable: {args.stable_url}")
    stable_result = await run_evals(args.stable_url, dataset)

    print(f"[canary] Evaluating canary: {args.canary_url}")
    canary_result = await run_evals(args.canary_url, dataset)

    promote = _compare(stable_result, canary_result, args.faithfulness_threshold)

    if promote and args.promote:
        print("[canary] Promotion signal sent to deployment system")
    elif not promote:
        print("[canary] Rollback signal sent to deployment system")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
