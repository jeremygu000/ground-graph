"""Ontology loader and validator.

Loads the versioned YAML ontology and provides validation
helpers. The ontology defines entity types and predicates
that constrain what can be stored in the knowledge graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ONTOLOGY_DIR = Path(
    os.environ.get(
        "GROUNDGRAPH_ONTOLOGY_DIR",
        str(Path(__file__).parent.parent.parent.parent.parent / "ontology"),
    )
)


@dataclass
class EntityTypeAttribute:
    name: str
    type: str
    required: bool = False


@dataclass
class EntityType:
    name: str
    description: str
    attributes: list[EntityTypeAttribute] = field(default_factory=list["EntityTypeAttribute"])

    def has_attribute(self, name: str) -> bool:
        return any(a.name == name for a in self.attributes)


@dataclass
class Predicate:
    name: str
    description: str
    domain: list[str]
    range: list[str]
    symmetric: bool = False


@dataclass
class Constraint:
    description: str
    expression: str


@dataclass
class Ontology:
    version: str
    entity_types: list[EntityType] = field(default_factory=list["EntityType"])
    predicates: list[Predicate] = field(default_factory=list["Predicate"])
    constraints: list[Constraint] = field(default_factory=list["Constraint"])

    def get_entity_type(self, name: str) -> EntityType | None:
        for et in self.entity_types:
            if et.name == name:
                return et
        return None

    def get_predicate(self, name: str) -> Predicate | None:
        for p in self.predicates:
            if p.name == name:
                return p
        return None

    def is_valid_predicate(self, predicate: str) -> bool:
        return self.get_predicate(predicate) is not None

    def is_valid_entity_type(self, entity_type: str) -> bool:
        return self.get_entity_type(entity_type) is not None


def _parse_entity_type(raw: dict[str, Any]) -> EntityType:
    attrs: list[EntityTypeAttribute] = []
    for a in raw.get("attributes", []):
        attrs.append(
            EntityTypeAttribute(
                name=str(a["name"]),
                type=str(a["type"]),
                required=bool(a.get("required", False)),
            )
        )
    return EntityType(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        attributes=attrs,
    )


def _parse_predicate(raw: dict[str, Any]) -> Predicate:
    return Predicate(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        domain=[str(d) for d in raw.get("domain", [])],
        range=[str(r) for r in raw.get("range", [])],
        symmetric=bool(raw.get("symmetric", False)),
    )


def _parse_constraint(raw: dict[str, Any]) -> Constraint:
    return Constraint(
        description=str(raw.get("description", "")),
        expression=str(raw["expression"]),
    )


def load_ontology(version: str | None = None) -> Ontology:
    """Load an ontology version from YAML.

    If *version* is None, loads the latest available version.
    """
    if version is None:
        version = _get_latest_version()

    entity_types_path = _ONTOLOGY_DIR / "entity_types.yaml"
    if not entity_types_path.exists():
        raise FileNotFoundError(f"Ontology not found at {entity_types_path}")

    with open(entity_types_path) as f:
        raw = yaml.safe_load(f)

    if raw.get("version") != version:
        raise ValueError(f"Ontology version mismatch: expected {version}, got {raw.get('version')}")

    entity_types = [_parse_entity_type(et) for et in raw.get("entity_types", [])]
    predicates = [_parse_predicate(p) for p in raw.get("predicates", [])]
    constraints = [_parse_constraint(c) for c in raw.get("constraints", [])]

    return Ontology(
        version=version,
        entity_types=entity_types,
        predicates=predicates,
        constraints=constraints,
    )


def _get_latest_version() -> str:
    versions_dir = _ONTOLOGY_DIR / "versions"
    if not versions_dir.exists():
        return "v0.1.0"
    files = sorted(versions_dir.glob("*.yaml"))
    if not files:
        return "v0.1.0"
    return files[-1].stem


_ontology_cache: Ontology | None = None


def get_ontology() -> Ontology:
    """Return the cached singleton ontology."""
    global _ontology_cache  # noqa: PLW0603
    if _ontology_cache is None:
        _ontology_cache = load_ontology()
    return _ontology_cache
