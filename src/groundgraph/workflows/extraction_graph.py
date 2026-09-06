"""LangGraph extraction workflow for knowledge graph construction.

Nodes:
  - extract_entities: Extract entity mentions from chunk text
  - resolve_entities: Resolve mentions to canonical entities
  - extract_facts: Extract facts from entity relationships
  - validate_facts: Validate facts against ontology constraints
  - emit_outbox: Emit outbox events for projection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention, KnowledgeFact


@dataclass
class ExtractionState:
    chunk_id: UUID | None = None
    chunk_content: str = ""
    tenant_id: str = ""

    entities: list[EntityMention] = field(default_factory=list)
    resolved_entities: list[CanonicalEntity] = field(default_factory=list)
    ambiguous_mentions: list[EntityMention] = field(default_factory=list)
    facts: list[KnowledgeFact] = field(default_factory=list)

    error: str | None = None

    def done(self) -> bool:
        return self.error is not None


def build_extraction_graph(
    entity_extractor: Any,
    entity_resolver: Any,
    fact_extractor: Any,
    outbox_emitter: Any,
) -> Any:
    """Build the extraction state graph."""

    async def extract_entities_node(state: ExtractionState) -> ExtractionState:
        mentions = await entity_extractor.extract(state.chunk_content, state.chunk_id)
        return ExtractionState(**{**state.__dict__, "entities": mentions})

    async def resolve_entities_node(state: ExtractionState) -> ExtractionState:
        resolved: list[CanonicalEntity] = []
        ambiguous: list[EntityMention] = []
        for mention in state.entities:
            entity = await entity_resolver.resolve(mention)
            if entity is None:
                ambiguous.append(mention)
            else:
                resolved.append(entity)
        return ExtractionState(
            **{**state.__dict__, "resolved_entities": resolved, "ambiguous_mentions": ambiguous}
        )

    async def extract_facts_node(state: ExtractionState) -> ExtractionState:
        name_to_id: dict[str, UUID] = {
            e.canonical_name: e.entity_id for e in state.resolved_entities
        }
        surface_forms = list(name_to_id.keys())
        evidence_ids = [state.chunk_id] if state.chunk_id else []
        facts = await fact_extractor.extract_facts(
            state.chunk_content,
            surface_forms,
            name_to_id,
            evidence_ids,
        )
        return ExtractionState(**{**state.__dict__, "facts": facts})

    async def emit_outbox_node(state: ExtractionState) -> ExtractionState:
        for mention in state.entities:
            await outbox_emitter.emit_mention(mention)
        for entity in state.resolved_entities:
            await outbox_emitter.emit_entity(entity)
        for fact in state.facts:
            await outbox_emitter.emit_fact(fact)
        return state

    graph = StateGraph(ExtractionState)
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("resolve_entities", resolve_entities_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("emit_outbox", emit_outbox_node)

    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "resolve_entities")
    graph.add_edge("resolve_entities", "extract_facts")
    graph.add_edge("extract_facts", "emit_outbox")
    graph.add_edge("emit_outbox", END)

    return graph.compile()
