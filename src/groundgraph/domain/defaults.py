"""Typed default factories for domain Pydantic models."""

from __future__ import annotations

from typing import Any

from groundgraph.domain.types import JsonValue


def empty_str_list() -> list[str]:
    return []


def empty_uuid_list() -> list[Any]:
    return []


def empty_json_dict() -> dict[str, JsonValue]:
    return {}


def empty_resolved_entity_list() -> list[Any]:
    return []


def empty_citation_list() -> list[Any]:
    return []


def empty_answer_claim_list() -> list[Any]:
    return []
