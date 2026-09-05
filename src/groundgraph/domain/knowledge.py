"""Knowledge domain contracts (plan.md §4.2).

Pure Pydantic v2 — no infrastructure imports.

These are the contracts that get stored in the knowledge graph
(Neo4j) and projected from the outbox. Provenance, validity
intervals, and confidence are first-class fields (see ADR-003).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from groundgraph.domain.defaults import empty_json_dict, empty_str_list, empty_uuid_list
from groundgraph.domain.types import JsonValue


class EntityMention(BaseModel):
    """A surface form of an entity spotted inside a chunk.

    Mentions are written first (cheap) and then resolved to
    CanonicalEntity by the entity-resolution workflow (M5+).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mention_id: UUID
    chunk_id: UUID
    surface_form: str
    candidate_type: str
    locator: str | None = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class CanonicalEntity(BaseModel):
    """The resolved, deduplicated form of one real-world entity.

    ``canonical_name`` is the display name. ``aliases`` are the
    surface forms (from ``EntityMention``) that resolve to this
    entity; the knowledge graph builds an index over them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: UUID
    entity_type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=empty_str_list)
    attributes: dict[str, JsonValue] = Field(default_factory=empty_json_dict)


class KnowledgeFact(BaseModel):
    """A reified, dated, provenance-bearing triple.

    Reification is required so the system can express:
      * confidence (we are not 100% sure);
      * validity intervals (the fact was true from X to Y, even
        if it was observed at Z);
      * supersession (a newer fact with the same subject+predicate
        invalidates the old one without losing history);
      * provenance chain (which evidence chunks back the fact).

    See ADR-003 for the rationale.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_id: UUID
    subject_id: UUID
    predicate: str
    object_id: UUID
    status: Literal["candidate", "verified", "rejected", "superseded"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[UUID] = Field(default_factory=empty_uuid_list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime
    extraction_method: Literal["structured", "rule", "llm", "human"]
    ontology_version: str
