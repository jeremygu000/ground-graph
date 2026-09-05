"""Shared primitive types for the domain layer.

Domain contracts across plan.md §4 use these building blocks. The
domain layer is pure Pydantic + stdlib — no infrastructure imports
allowed (see AGENTS.md §4).
"""

from __future__ import annotations

import math
from typing import TypeAlias


class FrozenJsonList(list[object]):
    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON containers are immutable once validated")

    append = extend = insert = pop = remove = clear = sort = reverse = _immutable
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable


class FrozenJsonDict(dict[str, object]):
    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("JSON containers are immutable once validated")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


JsonValue: TypeAlias = bool | int | float | str | None | FrozenJsonList | FrozenJsonDict


def validate_json_value(value: object) -> object:
    """Recursively validate that ``value`` is JSON-serialisable.

    Raises ``ValueError`` for non-JSON types (class instances, datetime,
    UUID, bytes, NaN/Infinity) or non-string dict keys.
    """
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("JSON float must not be NaN or Infinity")
        return value
    if isinstance(value, list):
        items = FrozenJsonList(validate_json_value(item) for item in value)
        return items
    if isinstance(value, dict):
        mapping = FrozenJsonDict()
        for k, v in value.items():
            if not isinstance(k, str):
                msg = "JSON object key must be str, got " + type(k).__name__
                raise TypeError(msg)
            dict.__setitem__(mapping, k, validate_json_value(v))
        return mapping
    raise ValueError(
        f"JsonValue must be str | int | float | bool | None | list | dict, "
        f"got {type(value).__name__}"
    )
