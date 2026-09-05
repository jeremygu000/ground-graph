"""Shared primitive types for the domain layer.

Domain contracts across plan.md §4 use these building blocks. The
domain layer is pure Pydantic + stdlib — no infrastructure imports
allowed (see AGENTS.md §4).
"""

from __future__ import annotations

import math
from typing import cast

type JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


def validate_json_value(value: object) -> JsonValue:
    """Recursively validate that ``value`` is JSON-serialisable.

    Raises ``ValueError`` for non-JSON types (class instances, datetime,
    UUID, bytes, NaN/Infinity) or non-string dict keys.

    Immutability is guaranteed by the domain model being ``frozen=True``;
    this function only guarantees JSON-structure correctness.
    """
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("JSON float must not be NaN or Infinity")
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        return [validate_json_value(item) for item in items]
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        result: dict[str, JsonValue] = {}
        for k, v in mapping.items():
            if not isinstance(k, str):
                msg = "JSON object key must be str, got " + type(k).__name__
                raise TypeError(msg)
            result[k] = validate_json_value(v)
        return result
    raise ValueError(
        f"JsonValue must be str | int | float | bool | None | list | dict, "
        f"got {type(value).__name__}"
    )
