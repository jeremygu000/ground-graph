"""Reciprocal Rank Fusion for hybrid vector + keyword retrieval.

Pure data fusion — accepts application-layer :class:`RetrievedChunk`
values from both retrievers and returns fused, ranked chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groundgraph.application.ports import RetrievedChunk

RRF_K = 60


@dataclass
class FusionResult:
    chunk_id: Any
    source_id: Any
    document_id: Any | None
    version_id: Any | None
    content: str
    vector_score: float | None
    keyword_score: float | None
    rrf_score: float
    allowed_principals: list[str]


def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    *,
    k: int = RRF_K,
) -> list[FusionResult]:
    """Combine vector and keyword rankings using Reciprocal Rank Fusion.

    RRF formula:  score(d) = Σ  1 / (k + rank(d))
    """
    rrf_scores: dict[Any, dict[str, Any]] = {}

    for rank, vec_result in enumerate(vector_results, start=1):
        cid = vec_result.chunk_id
        if cid not in rrf_scores:
            rrf_scores[cid] = {
                "chunk_id": cid,
                "source_id": vec_result.source_id,
                "document_id": vec_result.document_id,
                "version_id": vec_result.version_id,
                "content": vec_result.content,
                "vector_score": vec_result.vector_score,
                "keyword_score": None,
                "allowed_principals": list(vec_result.allowed_principals),
            }
        rrf_scores[cid]["rrf_score"] = rrf_scores[cid].get("rrf_score", 0.0) + (1 / (k + rank))
        rrf_scores[cid]["vector_score"] = vec_result.vector_score

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
                "keyword_score": kw_result.keyword_score,
                "allowed_principals": list(kw_result.allowed_principals),
            }
        rrf_scores[cid]["rrf_score"] = rrf_scores[cid].get("rrf_score", 0.0) + (1 / (k + rank))
        rrf_scores[cid]["keyword_score"] = kw_result.keyword_score

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
