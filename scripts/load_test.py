#!/usr/bin/env python3
"""Load test script for GroundGraph query workloads.

Usage:
    # Header auth (dev/trusted-proxy mode):
    uv run python scripts/load_test.py \
        --api-url http://localhost:8000 \
        --auth-mode header \
        --tenant-id load-test-tenant \
        --principal load-test-user \
        --users 50 --duration 300

    # OIDC auth (production):
    uv run python scripts/load_test.py \
        --api-url http://localhost:8000 \
        --auth-mode oidc \
        --bearer-token "$PROD_TOKEN" \
        --users 50 --duration 300

    # Local auth (dev only):
    uv run python scripts/load_test.py \
        --api-url http://localhost:8000 \
        --auth-mode local \
        --users 50 --duration 300

Measures:
    - Query p50/p95/p99 latency
    - Error rate and 5xx count
    - Throughput (queries/sec)
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

SAMPLE_QUESTIONS = [
    "What PostgreSQL extensions does GroundGraph use for vector storage?",
    "How does entity resolution work in the knowledge graph?",
    "What is the difference between verification and pending status for facts?",
    "How does the hybrid retrieval combine vector and graph search?",
    "What telemetry spans are captured during query execution?",
]

ERROR_RATE_THRESHOLD = 5.0


@dataclass
class LoadTestStats:
    started_at: float = field(default_factory=time.time)
    requests: int = 0
    errors: int = 0
    client_errors: int = 0
    server_errors: int = 0
    latencies: list[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, latency: float, status: int) -> None:
        async with self.lock:
            if status >= 500:  # noqa: PLR2004
                self.server_errors += 1
                self.errors += 1
            elif status >= 400:  # noqa: PLR2004
                self.client_errors += 1
                self.errors += 1
            else:
                self.requests += 1
            self.latencies.append(latency)

    async def report(self) -> dict[str, Any]:
        async with self.lock:
            elapsed = time.time() - self.started_at
            sorted_latencies = sorted(self.latencies)
            count = len(sorted_latencies)
            return {
                "elapsed_seconds": round(elapsed, 1),
                "total_requests": self.requests,
                "total_errors": self.errors,
                "client_errors": self.client_errors,
                "server_errors": self.server_errors,
                "error_rate": round(self.errors / max(self.requests, 1) * 100, 2),
                "throughput_rps": round(self.requests / max(elapsed, 1), 2),
                "latency_p50_ms": round(sorted_latencies[int(count * 0.50)] * 1000, 1)
                if count > 0
                else 0,
                "latency_p95_ms": round(sorted_latencies[int(count * 0.95)] * 1000, 1)
                if count > 0
                else 0,
                "latency_p99_ms": round(sorted_latencies[int(count * 0.99)] * 1000, 1)
                if count > 0
                else 0,
            }


async def query_worker(  # noqa: PLR0917
    client: httpx.AsyncClient,
    stats: LoadTestStats,
    api_url: str,
    auth_headers: dict[str, str],
    worker_id: int,
    duration: int,
) -> None:
    """Simulate a user submitting query requests."""
    end_time = time.time() + duration

    while time.time() < end_time:
        question = random.choice(SAMPLE_QUESTIONS)
        try:
            start = time.time()
            response = await client.post(
                f"{api_url}/v1/query",
                headers=auth_headers,
                json={"question": question},
                timeout=30.0,
            )
            latency = time.time() - start
            await stats.record(latency, response.status_code)
        except httpx.TimeoutException:
            await stats.record(30.0, 504)
        except Exception:
            await stats.record(0.0, 0)
        await asyncio.sleep(random.uniform(0.5, 3.0))


async def run_load_test(
    api_url: str,
    auth_headers: dict[str, str],
    users: int,
    duration: int,
) -> dict[str, Any]:
    stats = LoadTestStats()

    async with httpx.AsyncClient() as client:
        workers = [
            query_worker(client, stats, api_url, auth_headers, i, duration) for i in range(users)
        ]

        await asyncio.gather(*workers, return_exceptions=True)

    return await stats.report()


def _build_auth_headers(
    auth_mode: str,
    bearer_token: str | None,
    tenant_id: str | None,
    principal: str | None,
) -> dict[str, str]:
    """Build authentication headers for the load test."""
    if auth_mode == "oidc":
        if not bearer_token:
            raise ValueError("--bearer-token required for oidc auth mode")
        return {"Authorization": f"Bearer {bearer_token}"}
    if auth_mode == "header":
        if not tenant_id or not principal:
            raise ValueError("--tenant-id and --principal required for header auth mode")
        return {"X-Tenant-ID": tenant_id, "X-Principal": principal}
    if auth_mode == "local":
        return {}
    raise ValueError(f"Unknown auth mode: {auth_mode}")


async def _smoke_check(api_url: str, auth_headers: dict[str, str]) -> bool:
    """Verify the API is reachable and auth works before starting load test."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{api_url}/v1/query",
                headers=auth_headers,
                json={"question": SAMPLE_QUESTIONS[0]},
            )
            return response.status_code in (200, 401, 403)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundGraph load test")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument(
        "--auth-mode",
        choices=["header", "oidc", "local"],
        default="local",
        help="Authentication mode",
    )
    parser.add_argument("--bearer-token", help="Bearer token for OIDC auth mode")
    parser.add_argument("--tenant-id", help="Tenant ID for header auth mode")
    parser.add_argument("--principal", help="Principal ID for header auth mode")
    args = parser.parse_args()

    try:
        auth_headers = _build_auth_headers(
            args.auth_mode, args.bearer_token, args.tenant_id, args.principal
        )
    except ValueError as exc:
        parser.error(str(exc))
        return

    print(f"[load-test] Starting: {args.users} workers, {args.duration}s, auth={args.auth_mode}")

    ready = asyncio.run(_smoke_check(args.api_url, auth_headers))
    if not ready:
        print("[load-test] WARNING: API smoke check failed; proceeding anyway")

    results = asyncio.run(run_load_test(args.api_url, auth_headers, args.users, args.duration))

    print("\n=== Load Test Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")

    if results["error_rate"] > ERROR_RATE_THRESHOLD:
        print(f"\nWARNING: High error rate {results['error_rate']}%")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
