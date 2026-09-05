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
    Raises ``ValueError`` for cyclic container references.

    The returned containers are plain ``list``/``dict`` — immutability is
    NOT guaranteed at runtime.  The surrounding Pydantic model is
    ``frozen=True`` which prevents *field replacement*, but does not
    prevent in-place mutation of nested mutable containers.
    """
    return _validate(value, set())


def _validate(value: object, seen: set[int]) -> JsonValue:
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("JSON float must not be NaN or Infinity")
        return value
    if isinstance(value, list):
        lst = cast(list[object], value)
        object_id = id(lst)
        if object_id in seen:
            raise ValueError("JSON list contains a cycle")
        seen.add(object_id)
        try:
            return [_validate(item, seen) for item in lst]
        finally:
            seen.discard(object_id)
    if isinstance(value, dict):
        dct = cast(dict[object, object], value)
        object_id = id(dct)
        if object_id in seen:
            raise ValueError("JSON dict contains a cycle")
        seen.add(object_id)
        try:
            mapping = cast(dict[object, object], value)
            result: dict[str, JsonValue] = {}
            for k, v in mapping.items():
                if not isinstance(k, str):
                    msg = "JSON object key must be str, got " + type(k).__name__
                    raise TypeError(msg)
                result[k] = _validate(v, seen)
            return result
        finally:
            seen.discard(object_id)
    raise ValueError(
        f"JsonValue must be str | int | float | bool | None | list | dict, "
        f"got {type(value).__name__}"
    )
