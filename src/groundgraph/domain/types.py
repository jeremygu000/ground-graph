"""Shared primitive types for the domain layer.

Domain contracts across plan.md §4 use these building blocks. The
domain layer is pure Pydantic + stdlib — no infrastructure imports
allowed (see AGENTS.md §4).
"""

from __future__ import annotations

from typing import Any

JsonValue = Any
"""Recursive JSON value: str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue].

Pydantic v2 accepts ``Any`` as a permissive container and serialises
nested structures correctly. The contract is structural, not nominal:
* dict keys must be ``str``;
* list items are themselves ``JsonValue``;
* no custom classes pass through.
"""
