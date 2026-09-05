"""Shared primitive types for the domain layer.

Domain contracts across plan.md §4 use these building blocks. The
domain layer is pure Pydantic + stdlib — no infrastructure imports
allowed (see AGENTS.md §4).
"""

from __future__ import annotations

import math
from typing import cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
"""Recursive JSON value — validated to reject non-JSON types at domain boundary."""


def validate_json_value(value: object) -> JsonValue:
    """Recursively validate that ``value`` is JSON-serialisable.

    Raises ``ValueError`` for non-JSON types (class instances, datetime,
    UUID, bytes, NaN/Infinity) or non-string dict keys.
    """
    if isinstance(value, (str, int, bool, type(None))):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("JSON float must not be NaN or Infinity")
        return cast(JsonValue, value)
    if isinstance(value, list):
        items = cast(list[object], value)
        validated_items: list[JsonValue] = []
        for item in items:
            validated_items.append(validate_json_value(item))
        return validated_items
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        validated_mapping: dict[str, JsonValue] = {}
        for k, v in mapping.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"JSON object key must be str, got {type(k).__name__}"
                )
            validated_mapping[k] = validate_json_value(v)
        return validated_mapping
    raise ValueError(
        f"JsonValue must be str | int | float | bool | None | list | dict, "
        f"got {type(value).__name__}"
    )
