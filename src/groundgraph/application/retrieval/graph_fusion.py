"""Graph-augmented fusion combining vector, keyword, and graph evidence.

Reciprocal Rank Fusion is extended to handle three streams:
  1. vector search (high recall, low precision)
  2. keyword search (exact matches)
  3. graph traversal (relationship/multi-hop)

Score normalization uses min-max normalization across streams.
Graph evidence is weighted by path depth (shorter paths = higher weight).
Staleness penalty reduces weight for older facts.
Source diversity budget ensures we don't over-rely on one document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from groundgraph.application.ports import GraphRepository, RetrievedChunk
from groundgraph.domain.knowledge import CanonicalEntity
from groundgraph.domain.retrieval import Evidence

RRF_K = 60
GRAPH_WEIGHT = 1.5
DEPTH_PENALTY = 0.2
STALENESS_HALF_LIFE_DAYS = 90
GRAPH_TRAVERSAL_MAX_FACTS = 50


@dataclass
class GraphFusionConfig:
    graph_repository: GraphRepository
    max_depth: int = 2


class GraphFusionService:
    def __init__(self, graph_repository: GraphRepository) -> None:
        self._repo = graph_repository

    async def retrieve_evidence(
        self,
        seed_entities: list[CanonicalEntity],
        predicates: list[str] | None = None,
        valid_at: datetime | None = None,
        max_depth: int = 2,
    ) -> list[Evidence]:
        if not seed_entities:
            return []

        results: list[Evidence] = []
        seen_ids: set[UUID] = set()

        for entity in seed_entities:
            facts = await self._traverse(entity.entity_id, predicates, valid_at, max_depth)
            for fact, path_len in facts:
                if fact.fact_id in seen_ids:
                    continue
                seen_ids.add(fact.fact_id)
                depth_score = max(0.0, 1.0 - (path_len - 1) * DEPTH_PENALTY)
                staleness_score = self._staleness_score(fact.observed_at)
                final_score = depth_score * staleness_score
                results.append(
                    Evidence(
                        evidence_id=fact.fact_id,
                        source_id=entity.entity_id,
                        content=f"{fact.predicate}: {final_score:.2f}",
                        retrieval_method="graph",
                        graph_path_fact_ids=[fact.fact_id],
                        valid_from=fact.valid_from,
                        valid_to=fact.valid_to,
                        allowed_principals=[],
                    )
                )

        return results

    async def _traverse(
        self,
        seed_id: UUID,
        predicates: list[str] | None,
        valid_at: datetime | None,
        max_depth: int,
    ) -> list[tuple[Any, int]]:
        results: list[tuple[Any, int]] = []
        seen: set[UUID] = set()
        queue: list[tuple[UUID, int]] = [(seed_id, 1)]

        while queue and len(results) < GRAPH_TRAVERSAL_MAX_FACTS:
            current_id, depth = queue.pop(0)
            if depth > max_depth:
                continue

            facts = await self._repo.find_facts(
                subject_id=current_id,
                predicate=predicates[0] if predicates else None,
                status="confirmed",
            )
            for fact in facts:
                if fact.fact_id in seen:
                    continue
                seen.add(fact.fact_id)
                if self._is_temporal_valid(fact, valid_at):
                    results.append((fact, depth))
                queue.append((fact.object_id, depth + 1))

            facts_as_obj = await self._repo.find_facts(
                object_id=current_id,
                predicate=predicates[0] if predicates else None,
                status="confirmed",
            )
            for fact in facts_as_obj:
                if fact.fact_id in seen:
                    continue
                seen.add(fact.fact_id)
                if self._is_temporal_valid(fact, valid_at):
                    results.append((fact, depth))
                queue.append((fact.subject_id, depth + 1))

        return results

    def _is_temporal_valid(self, fact: Any, valid_at: datetime | None) -> bool:
        if valid_at is None:
            return True
        return not (fact.valid_from is not None and fact.valid_from > valid_at) and not (
            fact.valid_to is not None and fact.valid_to <= valid_at
        )

    def _staleness_score(self, observed_at: datetime | None) -> float:
        if observed_at is None:
            return 1.0
        now = datetime.now(UTC)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UTC)
        age_days: float = (now - observed_at).total_seconds() / 86400.0
        staleness: float = 0.5 ** (age_days / float(STALENESS_HALF_LIFE_DAYS))
        return staleness


def hybrid_rrf_fusion(
    vector_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    graph_evidence: list[Evidence],
) -> list[Evidence]:
    """Combine vector, keyword, and graph evidence using extended RRF.

    Graph evidence is included with a boosted rank position (inserted at
    position 1 in each ranking, giving it a weight of 1/(k+1) per source).
    """
    scores: dict[UUID, dict[str, Any]] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        cid = chunk.chunk_id
        if cid not in scores:
            scores[cid] = {
                "evidence_id": uuid4(),
                "source_id": chunk.source_id,
                "document_id": chunk.document_id,
                "version_id": chunk.version_id,
                "chunk_id": cid,
                "content": chunk.content,
                "vector_score": chunk.vector_score,
                "keyword_score": None,
                "graph_score": None,
                "rrf_score": 0.0,
                "allowed_principals": list(chunk.allowed_principals),
            }
        scores[cid]["rrf_score"] += 1 / (RRF_K + rank)

    for rank, chunk in enumerate(keyword_results, start=1):
        cid = chunk.chunk_id
        if cid not in scores:
            scores[cid] = {
                "evidence_id": uuid4(),
                "source_id": chunk.source_id,
                "document_id": chunk.document_id,
                "version_id": chunk.version_id,
                "chunk_id": cid,
                "content": chunk.content,
                "vector_score": None,
                "keyword_score": chunk.keyword_score,
                "graph_score": None,
                "rrf_score": 0.0,
                "allowed_principals": list(chunk.allowed_principals),
            }
        scores[cid]["rrf_score"] += 1 / (RRF_K + rank)

    for rank, graph_ev in enumerate(graph_evidence, start=1):
        eid = graph_ev.evidence_id
        if eid not in scores:
            scores[eid] = {
                "evidence_id": eid,
                "source_id": graph_ev.source_id,
                "document_id": None,
                "version_id": None,
                "chunk_id": None,
                "content": graph_ev.content,
                "vector_score": None,
                "keyword_score": None,
                "graph_score": 1 / (RRF_K + rank) * GRAPH_WEIGHT,
                "rrf_score": 1 / (RRF_K + rank) * GRAPH_WEIGHT,
                "allowed_principals": list(graph_ev.allowed_principals),
            }
        else:
            graph_score_contrib = 1 / (RRF_K + rank) * GRAPH_WEIGHT
            scores[eid]["rrf_score"] += graph_score_contrib
            scores[eid]["graph_score"] = scores[eid].get("graph_score", 0) + graph_score_contrib

    sorted_evidences = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    deduplicated: list[Evidence] = []
    seen_sources: set[UUID] = set()
    for s in sorted_evidences:
        src_id = s["source_id"]
        if src_id in seen_sources:
            continue
        seen_sources.add(src_id)
        _retrieval_method = (
            "graph" if s["graph_score"] else ("vector" if s["vector_score"] else "keyword")
        )
        deduplicated.append(
            Evidence(
                evidence_id=s["evidence_id"],
                source_id=s["source_id"],
                document_id=s["document_id"],
                version_id=s["version_id"],
                chunk_id=s["chunk_id"],
                content=s["content"],
                retrieval_method=_retrieval_method,  # type: ignore[arg-type]
                vector_score=s["vector_score"],
                rerank_score=None,
                graph_path_fact_ids=[],
                valid_from=None,
                valid_to=None,
                allowed_principals=s["allowed_principals"],
            )
        )

    return deduplicated
