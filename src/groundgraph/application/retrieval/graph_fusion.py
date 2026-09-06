"""Graph traversal and evidence fusion for the retrieval layer.

Application-layer service.  Depends only on application ports; the
infrastructure layer provides concrete GraphRepository implementations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import UUID, uuid4

from groundgraph.application.retrieval.fusion import RRF_K
from groundgraph.domain.knowledge import CanonicalEntity
from groundgraph.domain.retrieval import Evidence

if TYPE_CHECKING:
    from groundgraph.application.ports import GraphRepository


GRAPH_TRAVERSAL_MAX_FACTS = 50
DEPTH_PENALTY = 0.1
STALENESS_HALF_LIFE_DAYS = 30
GRAPH_WEIGHT = 1.5


class _FactWithNames(TypedDict):
    fact: Any
    depth: int
    subject_name: str
    object_name: str


class GraphFusionService:
    """Service for traversing the knowledge graph and retrieving graph-structured evidence.

    Graph traversal respects:
      - status = "verified" filter
      - temporal validity (valid_from / valid_to) against query valid_at
      - ACL via allowed_principals filter on facts
    """

    def __init__(self, graph_repository: GraphRepository) -> None:
        self._repo = graph_repository

    async def retrieve_evidence(  # noqa: PLR0917
        self,
        seed_entities: list[CanonicalEntity],
        predicates: list[str] | None = None,
        valid_at: datetime | None = None,
        max_depth: int = 2,
        tenant_id: str | None = None,
        principal: str | None = None,
    ) -> list[Evidence]:
        """Retrieve graph evidence traversing from seed entities.

        Args:
            seed_entities: starting entities for graph traversal
            predicates: optional list of predicates to filter edges
            valid_at: temporal point-in-time for temporal validity filtering
            max_depth: maximum traversal depth
            tenant_id: tenant identifier for ACL enforcement
            principal: requesting principal for ACL enforcement

        Returns:
            list of Evidence objects with hydrated content (subject predicate object)
        """
        if not seed_entities:
            return []

        results: list[Evidence] = []
        seen_ids: set[UUID] = set()
        allowed_principals = [principal] if principal else None

        for entity in seed_entities:
            facts = await self._traverse(
                entity.entity_id,
                predicates,
                valid_at,
                max_depth,
                allowed_principals,
            )
            for entry in facts:
                fact = entry["fact"]
                path_len = entry["depth"]
                subject_name = entry["subject_name"]
                obj_name = entry["object_name"]
                if fact.fact_id in seen_ids:
                    continue
                seen_ids.add(fact.fact_id)
                depth_score = max(0.0, 1.0 - (path_len - 1) * DEPTH_PENALTY)
                staleness_score = self._staleness_score(fact.observed_at)
                final_score = depth_score * staleness_score
                content = (
                    f"{subject_name} {fact.predicate} {obj_name} (confidence: {final_score:.2f})"
                )
                source_id = fact.evidence_ids[0] if fact.evidence_ids else fact.fact_id
                results.append(
                    Evidence(
                        evidence_id=fact.fact_id,
                        source_id=source_id,
                        content=content,
                        retrieval_method="graph",
                        vector_score=None,
                        rerank_score=None,
                        graph_path_fact_ids=[fact.fact_id],
                        valid_from=fact.valid_from,
                        valid_to=fact.valid_to,
                        allowed_principals=fact.allowed_principals,
                    )
                )

        return results

    async def _traverse(
        self,
        seed_id: UUID,
        predicates: list[str] | None,
        valid_at: datetime | None,
        max_depth: int,
        allowed_principals: list[str] | None,
    ) -> list[_FactWithNames]:
        results: list[_FactWithNames] = []
        seen: set[UUID] = set()
        queue: list[tuple[UUID, int]] = [(seed_id, 1)]

        while queue and len(results) < GRAPH_TRAVERSAL_MAX_FACTS:
            current_id, depth = queue.pop(0)
            if depth > max_depth:
                continue

            current_entity = await self._repo.get_entity(current_id)
            current_name = current_entity.canonical_name if current_entity else str(current_id)

            facts = await self._repo.find_facts(
                subject_id=current_id,
                predicate=predicates[0] if predicates else None,
                status="verified",
                allowed_principals=allowed_principals,
            )
            for fact in facts:
                if fact.fact_id in seen:
                    continue
                if not self._is_temporal_valid(fact, valid_at):
                    continue
                seen.add(fact.fact_id)
                obj_entity = await self._repo.get_entity(fact.object_id)
                obj_name = obj_entity.canonical_name if obj_entity else str(fact.object_id)
                results.append(
                    {
                        "fact": fact,
                        "depth": depth,
                        "subject_name": current_name,
                        "object_name": obj_name,
                    }
                )
                queue.append((fact.object_id, depth + 1))

            facts_as_obj = await self._repo.find_facts(
                object_id=current_id,
                predicate=predicates[0] if predicates else None,
                status="verified",
                allowed_principals=allowed_principals,
            )
            for fact in facts_as_obj:
                if fact.fact_id in seen:
                    continue
                if not self._is_temporal_valid(fact, valid_at):
                    continue
                seen.add(fact.fact_id)
                subj_entity = await self._repo.get_entity(fact.subject_id)
                subj_name = subj_entity.canonical_name if subj_entity else str(fact.subject_id)
                results.append(
                    {
                        "fact": fact,
                        "depth": depth,
                        "subject_name": subj_name,
                        "object_name": current_name,
                    }
                )
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


def _merge_temporal(
    scores: dict[UUID, dict[str, Any]],
    eid: UUID,
    graph_ev: Evidence,
    rank: int,
) -> None:
    graph_score_contrib = 1 / (RRF_K + rank) * GRAPH_WEIGHT
    scores[eid]["rrf_score"] += graph_score_contrib
    scores[eid]["graph_score"] = scores[eid].get("graph_score", 0) + graph_score_contrib
    scores[eid]["graph_path_fact_ids"] = list(
        set(scores[eid].get("graph_path_fact_ids", [])) | set(graph_ev.graph_path_fact_ids)
    )
    if graph_ev.valid_from is not None and (
        scores[eid]["valid_from"] is None or graph_ev.valid_from < scores[eid]["valid_from"]
    ):
        scores[eid]["valid_from"] = graph_ev.valid_from
    if graph_ev.valid_to is not None and (
        scores[eid]["valid_to"] is None or graph_ev.valid_to > scores[eid]["valid_to"]
    ):
        scores[eid]["valid_to"] = graph_ev.valid_to


def _determine_retrieval_method(s: dict[str, Any]) -> Literal["graph", "vector", "keyword"]:
    if s["graph_score"]:
        return "graph"
    if s["vector_score"]:
        return "vector"
    return "keyword"


def hybrid_rrf_fusion(
    vector_results: list[Any],
    keyword_results: list[Any],
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
                "graph_path_fact_ids": list(graph_ev.graph_path_fact_ids),
                "valid_from": graph_ev.valid_from,
                "valid_to": graph_ev.valid_to,
            }
        else:
            _merge_temporal(scores, eid, graph_ev, rank)

    sorted_evidences = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    deduplicated: list[Evidence] = []
    seen_sources: set[UUID] = set()
    for s in sorted_evidences:
        src_id = s["source_id"]
        if src_id in seen_sources:
            continue
        seen_sources.add(src_id)
        deduplicated.append(
            Evidence(
                evidence_id=s["evidence_id"],
                source_id=s["source_id"],
                document_id=s["document_id"],
                version_id=s["version_id"],
                chunk_id=s["chunk_id"],
                content=s["content"],
                retrieval_method=_determine_retrieval_method(s),
                vector_score=s["vector_score"],
                rerank_score=None,
                graph_path_fact_ids=s.get("graph_path_fact_ids", []),
                valid_from=s.get("valid_from"),
                valid_to=s.get("valid_to"),
                allowed_principals=s["allowed_principals"],
            )
        )

    return deduplicated
