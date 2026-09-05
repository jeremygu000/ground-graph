"""JSON snapshot helpers for PostgreSQL adapters."""

from __future__ import annotations

from typing import cast

from groundgraph.domain.types import validate_json_value


def snapshot_json_object(value: object) -> dict[str, object]:
    """Validate and copy a JSON object before persistence."""
    validated = validate_json_value(value)
    if not isinstance(validated, dict):
        raise TypeError(f"JSON object expected, got {type(validated).__name__}")
    return cast(dict[str, object], validated)
