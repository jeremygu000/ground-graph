"""Document domain contracts (plan.md §4.1).

Pure Pydantic v2 — no infrastructure imports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from groundgraph.domain.defaults import empty_json_dict, empty_str_list
from groundgraph.domain.types import JsonValue, validate_json_value


class SourceDescriptor(BaseModel):
    """Where a document came from and who is allowed to read it.

    ``allowed_principals`` is the authoritative access-control list
    for everything derived from this source. The retrieval layer
    filters on this BEFORE the model context is built (AGENTS.md §0.6
    rule 8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_id: UUID
    source_type: Literal["filesystem", "repository", "object_store", "api"]
    uri: str
    classification: str
    tenant_id: str
    allowed_principals: list[str] = Field(default_factory=empty_str_list)

    @model_validator(mode="after")
    def _reject_empty_tenant_id(self) -> SourceDescriptor:
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        return self


class ParsedDocument(BaseModel):
    """An immutable snapshot of a parsed document at one point in time.

    Documents are append-only: a re-ingest of the same source URI
    creates a NEW ``ParsedDocument`` with a new ``version_id`` and
    possibly a new ``document_id`` (if the source URI is new).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    document_id: UUID
    version_id: UUID
    source_id: UUID
    title: str
    media_type: str
    checksum: str
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=empty_json_dict)
    effective_at: datetime | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> object:
        return validate_json_value(value)

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return validate_json_value(value)  # type: ignore[return-value]


class Chunk(BaseModel):
    """A unit of text carved out of a ParsedDocument for retrieval.

    ``allowed_principals`` is duplicated from the parent document at
    chunk-create time so the retrieval filter is a single indexed
    column read, not a join back to the source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: UUID
    document_id: UUID
    version_id: UUID
    ordinal: int = Field(ge=0)
    heading_path: list[str] = Field(default_factory=empty_str_list)
    content: str
    token_count: int = Field(ge=0)
    checksum: str
    start_locator: str | None = None
    end_locator: str | None = None
    allowed_principals: list[str] = Field(default_factory=empty_str_list)
