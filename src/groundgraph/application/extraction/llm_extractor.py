"""LLM-based entity extractor using structured output."""

from __future__ import annotations

import json
from typing import cast
from uuid import UUID, uuid4

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from groundgraph.application.ports import EntityExtractor
from groundgraph.application.settings import Settings, get_settings
from groundgraph.domain.knowledge import EntityMention

_MIN_CONFIDENCE = 0.5


class _EntityOutput(BaseModel):
    surface_form: str
    entity_type: str
    confidence: float


class _ExtractionOutput(BaseModel):
    entities: list[_EntityOutput]


SYSTEM_PROMPT = (
    "You are an entity extraction assistant. Given the text, extract all named entities "
    "that match the following types: Person, SoftwareSystem, Repository, Service, Database, "
    "Configuration, Document, Concept.\n\n"
    "Return a JSON object with an 'entities' array. Each entity must have:\n"
    "- surface_form: the exact text of the mention\n"
    "- entity_type: one of the valid types\n"
    "- confidence: a number between 0 and 1\n\n"
    "Only extract entities you are confident about."
)


class LLMEntityExtractor(EntityExtractor):
    """LLM-based entity extractor for prose text."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._client = client or AsyncOpenAI(api_key=cfg.openai_api_key_value)
        self._model = cfg.generation_model

    async def extract(self, text: str, chunk_id: UUID) -> list[EntityMention]:
        """Extract entities using LLM structured output."""
        if not text.strip():
            return []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:4000]},
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
            output = _ExtractionOutput.model_validate(data)
        except Exception:
            return []

        results: list[EntityMention] = []
        for entity in output.entities:
            if entity.confidence < _MIN_CONFIDENCE:
                continue
            mention = EntityMention(
                mention_id=uuid4(),
                chunk_id=chunk_id,
                surface_form=entity.surface_form,
                candidate_type=entity.entity_type,
                extraction_confidence=entity.confidence,
            )
            results.append(mention)

        return results
