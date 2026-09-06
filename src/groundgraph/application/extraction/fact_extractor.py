"""Fact extraction from entity relationships.

Extracts structured facts (subject-predicate-object triples) from
document content using the ontology predicates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.knowledge import KnowledgeFact
from groundgraph.domain.ontology.loader import get_ontology

_MIN_CONFIDENCE = 0.5


class _FactOutput(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float


class _FactsOutput(BaseModel):
    facts: list[_FactOutput]


SYSTEM_PROMPT = (
    "You are a fact extraction assistant. Given a document and a list of entities, "
    "extract factual relationships (triples: subject-predicate-object).\n\n"
    "Valid predicates: depends_on, implements, deploys, documents, owns, configures, "
    "references, supersedes\n"
    "Return a JSON object with a 'facts' array. Each fact must have:\n"
    "- subject: the exact surface form of the subject entity\n"
    "- predicate: one of the valid predicates\n"
    "- object: the exact surface form of the object entity\n"
    "- confidence: a number between 0 and 1"
)


class FactExtractor:
    """Extract facts from text using LLM structured output."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._client = client or AsyncOpenAI(api_key=cfg.openai_api_key_value)
        self._model = cfg.generation_model
        self._ontology = get_ontology()

    async def extract_facts(
        self,
        text: str,
        entities: list[str],
        name_to_id: dict[str, UUID],
        evidence_ids: list[UUID],
        ontology_version: str | None = None,
        *,
        tenant_id: str | None = None,
        allowed_principals: list[str] | None = None,
    ) -> list[KnowledgeFact]:
        """Extract facts from text given entity surface forms and their IDs.

        name_to_id maps canonical surface forms to resolved entity IDs.
        Facts whose subject or object cannot be mapped are silently dropped.
        """
        if not text.strip() or not entities:
            return []

        valid_predicates = [p.name for p in self._ontology.predicates]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Entities: {', '.join(entities)}\n\n"
                    f"Valid predicates: {', '.join(valid_predicates)}\n\n"
                    f"Document:\n{text[:4000]}"
                ),
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception:
            return []

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            output = _FactsOutput.model_validate(data)
        except Exception:
            return []

        facts: list[KnowledgeFact] = []
        now = datetime.now(UTC)
        version = ontology_version or self._ontology.version

        for f in output.facts:
            if f.confidence < _MIN_CONFIDENCE:
                continue
            if f.predicate not in valid_predicates:
                continue
            sub_id = name_to_id.get(f.subject)
            obj_id = name_to_id.get(f.object)
            if sub_id is None or obj_id is None:
                continue
            fact = KnowledgeFact(
                fact_id=uuid4(),
                subject_id=sub_id,
                predicate=f.predicate,
                object_id=obj_id,
                status="candidate",
                confidence=f.confidence,
                evidence_ids=evidence_ids,
                observed_at=now,
                extraction_method="llm",
                ontology_version=version,
                tenant_id=tenant_id or "",
                allowed_principals=allowed_principals or [],
            )
            facts.append(fact)

        return facts
