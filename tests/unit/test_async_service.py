"""Async service test scaffold for M0.

This test demonstrates the async test pattern that will be used across
the application layer in later milestones.
"""

from __future__ import annotations

import asyncio

import pytest


async def ping() -> str:
    await asyncio.sleep(0)
    return "pong"


async def echo(value: str) -> str:
    return value


async def test_async_ping() -> None:
    assert await ping() == "pong"


async def test_async_echo() -> None:
    assert await echo("hello") == "hello"


async def test_async_gather() -> None:
    results = await asyncio.gather(ping(), echo("a"), echo("b"))
    assert results == ["pong", "a", "b"]


@pytest.mark.parametrize("value", ["a", "b", "c"])
async def test_async_param(value: str) -> None:
    assert (await echo(value)) == value
