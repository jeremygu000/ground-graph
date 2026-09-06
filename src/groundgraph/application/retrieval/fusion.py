"""Reciprocal Rank Fusion for hybrid vector + keyword retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groundgraph.infrastructure.postgres.keyword_retriever import KeywordSearchResult
from groundgraph.infrastructure.postgres.vector_retriever import VectorSearchResult

RRF_K = 60


@dataclass
class FusionResult:
    chunk_id: Any
    source_id: Any
    document_id: Any
    version_id: Any
    content: str
    vector_score: float | None
    keyword_score: float | None
    rrf_score: float
    allowed_principals: list[str]


def reciprocal_rank_fusion(
    vector_results: list[VectorSearchResult],
    keyword_results: list[KeywordSearchResult],
    *,
    k: int = RRF_K,
) -> list[FusionResult]:
    """Combine vector and keyword rankings using Reciprocal Rank Fusion.

    RRF formula:  score(d) = Σ  1 / (k + rank(d))

    Parameters
    ----------
    vector_results
        Pre-sorted vector results (best rank = 1).
    keyword_results
        Pre-sorted keyword results (best rank = 1).
    k
        RRF damping constant (default 60 per Gretarsson et al.).

    Returns
    -------
    list[FusionResult]
        Chunks sorted by descending RRF score.
    """
    rrf_scores: dict[Any, dict[str, Any]] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "chunk_id": cid,
                "source_id": result.source_id,
                "document_id": result.document_id,
                "version_id": result.version_id,
                "content": result.content,
                "vector_score": result.score,
                "keyword_score": None,
                "allowed_principals": result.allowed_principals,
            }
        rrf_scores[cid]["rrf_score"] = rrf_scores[cid].get("rrf_score", 0.0) + (1 / (k + rank))
        rrf_scores[cid]["vector_score"] = result.score

    for rank, kw_result in enumerate(keyword_results, start=1):
        cid = kw_result.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "chunk_id": cid,
                "source_id": kw_result.source_id,
                "document_id": kw_result.document_id,
                "version_id": kw_result.version_id,
                "content": kw_result.content,
                "vector_score": None,
                "keyword_score": kw_result.rank,
                "allowed_principals": kw_result.allowed_principals,
            }
        rrf_scores[cid]["rrf_score"] = rrf_scores[cid].get("rrf_score", 0.0) + (1 / (k + rank))
        rrf_scores[cid]["keyword_score"] = kw_result.rank

    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return [
        FusionResult(
            chunk_id=r["chunk_id"],
            source_id=r["source_id"],
            document_id=r["document_id"],
            version_id=r["version_id"],
            content=r["content"],
            vector_score=r["vector_score"],
            keyword_score=r["keyword_score"],
            rrf_score=r["rrf_score"],
            allowed_principals=r["allowed_principals"],
        )
        for r in sorted_results
    ]
