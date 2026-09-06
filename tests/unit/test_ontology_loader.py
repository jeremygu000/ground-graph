"""Unit tests for the ontology loader."""

from __future__ import annotations

from groundgraph.domain.ontology.loader import (
    get_ontology,
    load_ontology,
)


def test_load_ontology_returns_valid_ontology() -> None:
    """Ontology loads successfully and has expected structure."""
    ont = load_ontology("v0.1.0")
    assert ont.version == "v0.1.0"
    assert len(ont.entity_types) > 0
    assert len(ont.predicates) > 0


def test_ontology_has_person_entity_type() -> None:
    """Person entity type is defined."""
    ont = load_ontology("v0.1.0")
    person = ont.get_entity_type("Person")
    assert person is not None
    assert person.description == "A human individual"


def test_ontology_has_depends_on_predicate() -> None:
    """depends_on predicate is defined."""
    ont = load_ontology("v0.1.0")
    dep = ont.get_predicate("depends_on")
    assert dep is not None
    assert dep.domain == ["SoftwareSystem", "Service"]
    assert dep.symmetric is False


def test_is_valid_entity_type() -> None:
    """is_valid_entity_type returns True for known types."""
    ont = load_ontology("v0.1.0")
    assert ont.is_valid_entity_type("Person") is True
    assert ont.is_valid_entity_type("SoftwareSystem") is True
    assert ont.is_valid_entity_type("NotAType") is False


def test_is_valid_predicate() -> None:
    """is_valid_predicate returns True for known predicates."""
    ont = load_ontology("v0.1.0")
    assert ont.is_valid_predicate("depends_on") is True
    assert ont.is_valid_predicate("implements") is True
    assert ont.is_valid_predicate("not_a_predicate") is False


def test_get_ontology_caching() -> None:
    """get_ontology returns the same instance."""
    ont1 = get_ontology()
    ont2 = get_ontology()
    assert ont1 is ont2
