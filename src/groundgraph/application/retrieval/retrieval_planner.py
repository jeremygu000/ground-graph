"""Retrieval planner using LLM to generate explainable retrieval plans.

Given a natural-language question, the planner:
  1. Classifies the question type (fact/relationship/multi-hop/temporal/impact/...).
  2. Extracts entity seed hints from the question.
  3. Selects a retrieval strategy (vector/graph/hybrid).
  4. Emits reason codes explaining the strategy choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from groundgraph.application.ports import (
    EmbeddingProvider,
    EntityExtractor,
    EntityResolver,
)
from groundgraph.domain.ontology.loader import get_ontology
from groundgraph.domain.retrieval import ResolvedEntity, RetrievalPlan


@dataclass
class RetrievalPlannerConfig:
    entity_extractor: EntityExtractor
    entity_resolver: EntityResolver
    embedding_provider: EmbeddingProvider


class RetrievalPlanner:
    def __init__(self, config: RetrievalPlannerConfig) -> None:
        self._extractor = config.entity_extractor
        self._resolver = config.entity_resolver
        self._embed = config.embedding_provider

    async def plan(
        self,
        question: str,
        principal: str,
        tenant_id: str,
    ) -> RetrievalPlan:
        entities = await self._extract_entities(question, tenant_id)
        question_type = self._classify_question_type(question, entities)
        strategy = self._select_strategy(question_type, entities)
        reason_codes = self._build_reason_codes(question, question_type, strategy, entities)

        return RetrievalPlan(
            strategy=strategy,  # type: ignore[arg-type]
            question_type=question_type,  # type: ignore[arg-type]
            query_texts=[question],
            entities=entities,
            predicates=self._extract_predicates(question),
            max_graph_depth=self._graph_depth_for_type(question_type),
            vector_top_k=10,
            final_evidence_limit=10,
            valid_at=None,
            reason_codes=reason_codes,
        )

    async def _extract_entities(self, question: str, tenant_id: str) -> list[ResolvedEntity]:
        chunk_id = uuid4()
        mentions = await self._extractor.extract(question, chunk_id)
        resolved: list[ResolvedEntity] = []
        for mention in mentions:
            canonical = await self._resolver.resolve(mention)
            if canonical is not None:
                resolved.append(
                    ResolvedEntity(
                        entity_id=canonical.entity_id,
                        canonical_name=canonical.canonical_name,
                        entity_type=canonical.entity_type,
                    )
                )
        return resolved

    def _classify_question_type(self, question: str, entities: list[ResolvedEntity]) -> str:
        q = question.lower()
        multi_hop_threshold = 2
        temporal_kws = ["when", "before", "after", "history", "previously", "old version"]
        if any(kw in q for kw in temporal_kws):
            result = "temporal"
        else:
            rel_kws = ["depend", "impact", "affect", "connected to", "related to", "what services"]
            if any(kw in q for kw in rel_kws):
                result = "relationship"
            elif " vs " in q or " versus " in q or " compared " in q:
                result = "comparison"
            elif " what if " in q or " effect " in q or " would " in q:
                result = "impact"
            elif any(kw in q for kw in [" how ", " why ", " because "]):
                result = "fact"
            elif len(entities) >= multi_hop_threshold:
                result = "multi_hop"
            elif any(kw in q for kw in ["list", "all", "every", "which"]):
                result = "summary"
            else:
                result = "fact"
        return result

    def _select_strategy(self, question_type: str, entities: list[ResolvedEntity]) -> str:
        if question_type in ("relationship", "multi_hop", "impact", "temporal"):
            if entities:
                return "hybrid"
            return "graph"
        if question_type == "comparison":
            return "hybrid"
        return "vector"

    def _extract_predicates(self, question: str) -> list[str]:
        ontology = get_ontology()
        q = question.lower()
        found: list[str] = []
        for pred in ontology.predicates:
            pred_name_lower = pred.name.lower()
            if pred_name_lower in q:
                found.append(pred.name)
        return found

    def _graph_depth_for_type(self, question_type: str) -> int:
        depth_map = {
            "fact": 1,
            "relationship": 2,
            "multi_hop": 3,
            "temporal": 1,
            "comparison": 1,
            "impact": 2,
            "summary": 1,
            "unknown": 1,
        }
        return depth_map.get(question_type, 1)

    def _build_reason_codes(
        self,
        question: str,
        question_type: str,
        strategy: str,
        entities: list[ResolvedEntity],
    ) -> list[str]:
        codes: list[str] = []
        codes.append(f"q_type={question_type}")
        codes.append(f"strategy={strategy}")
        if entities:
            codes.append(f"entity_count={len(entities)}")
            entity_types = sorted({e.entity_type for e in entities})
            codes.append(f"entity_types={','.join(entity_types)}")
        if self._extract_predicates(question):
            codes.append("predicates_matched=true")
        return codes
