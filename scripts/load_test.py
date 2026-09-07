#!/usr/bin/env python3
"""Load test script for GroundGraph ingestion and query workloads.

Usage:
    uv run python scripts/load_test.py --api-url http://localhost:8000 --users 50 --duration 300

Measures:
    - Query p50/p95/p99 latency
    - Ingestion throughput (docs/sec)
    - Error rate
    - 5xx count
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

SAMPLE_QUESTIONS = [
    "What PostgreSQL extensions does GroundGraph use for vector storage?",
    "How does entity resolution work in the knowledge graph?",
    "What is the difference between verification and pending status for facts?",
    "How does the hybrid retrieval combine vector and graph search?",
    "What telemetry spans are captured during query execution?",
]

SAMPLE_DOCUMENT = {
    "content": "GroundGraph is a hybrid GraphRAG system that combines "
    "vector search with knowledge graph traversal for improved retrieval "
    "and answer generation.",
    "metadata": {
        "source": "load-test",
        "chunk_index": 0,
    },
}


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


async def query_worker(
    client: httpx.AsyncClient,
    stats: LoadTestStats,
    api_url: str,
    user_id: int,
    duration: int,
) -> None:
    """Simulate a user submitting query requests."""
    headers = {
        "X-Tenant-ID": "load-test-tenant",
        "X-Principal-ID": f"load-test-user-{user_id}",
    }
    end_time = time.time() + duration

    while time.time() < end_time:
        question = random.choice(SAMPLE_QUESTIONS)
        try:
            start = time.time()
            response = await client.post(
                f"{api_url}/v1/query",
                headers=headers,
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


async def ingestion_worker(
    client: httpx.AsyncClient,
    stats: LoadTestStats,
    api_url: str,
    worker_id: int,
    duration: int,
) -> None:
    """Simulate document ingestion."""
    headers = {
        "X-Tenant-ID": "load-test-tenant",
        "X-Principal-ID": f"load-test-ingest-{worker_id}",
    }
    end_time = time.time() + duration

    while time.time() < end_time:
        doc = {
            **SAMPLE_DOCUMENT,
            "metadata": {
                **SAMPLE_DOCUMENT["metadata"],
                "doc_id": str(uuid4()),
                "ingested_at": datetime.now(UTC).isoformat(),
            },
        }
        try:
            start = time.time()
            response = await client.post(
                f"{api_url}/v1/ingest",
                headers=headers,
                json=doc,
                timeout=60.0,
            )
            latency = time.time() - start
            await stats.record(latency, response.status_code)
        except httpx.TimeoutException:
            await stats.record(60.0, 504)
        except Exception:
            await stats.record(0.0, 0)
        await asyncio.sleep(random.uniform(1.0, 5.0))


async def run_load_test(
    api_url: str,
    users: int,
    query_ratio: float,
    duration: int,
) -> dict[str, Any]:
    stats = LoadTestStats()
    concurrency = users
    query_workers = max(1, int(concurrency * query_ratio))
    ingest_workers = max(1, concurrency - query_workers)

    async with httpx.AsyncClient() as client:
        workers = [
            query_worker(client, stats, api_url, i, duration) for i in range(query_workers)
        ] + [ingestion_worker(client, stats, api_url, i, duration) for i in range(ingest_workers)]

        await asyncio.gather(*workers, return_exceptions=True)

    return await stats.report()


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundGraph load test")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--query-ratio", type=float, default=0.8)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()

    print(f"[load-test] Starting load test: {args.users} users, {args.duration}s")
    print(f"[load-test] Query ratio: {args.query_ratio}, API: {args.api_url}")

    results = asyncio.run(run_load_test(args.api_url, args.users, args.query_ratio, args.duration))

    print("\n=== Load Test Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
