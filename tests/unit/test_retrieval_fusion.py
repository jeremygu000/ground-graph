"""Unit tests for RRF fusion."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.application.ports import RetrievedChunk
from groundgraph.application.retrieval.fusion import reciprocal_rank_fusion


def _vec_chunk(
    *,
    score: float | None = 0.1,
    content: str = "test content",
    chunk_id=None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid4(),
        source_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        content=content,
        vector_score=score,
        keyword_score=None,
        allowed_principals=["eng"],
    )


def _kw_chunk(
    *,
    rank: float | None = 0.05,
    content: str = "test content",
    chunk_id=None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid4(),
        source_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        content=content,
        vector_score=None,
        keyword_score=rank,
        allowed_principals=["eng"],
    )


class TestReciprocalRankFusion:
    def test_empty_results_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_vector_only_results(self) -> None:
        chunk_id = uuid4()
        result = reciprocal_rank_fusion([_vec_chunk(chunk_id=chunk_id, content="content a")], [])

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score == 0.1
        assert result[0].keyword_score is None
        assert result[0].rrf_score == pytest.approx(1 / 61)

    def test_keyword_only_results(self) -> None:
        chunk_id = uuid4()
        result = reciprocal_rank_fusion(
            [], [_kw_chunk(chunk_id=chunk_id, content="content a", rank=0.05)]
        )

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score is None
        assert result[0].keyword_score == 0.05

    def test_fusion_combines_both_sources(self) -> None:
        chunk_a = uuid4()
        chunk_b = uuid4()

        vector_results = [
            _vec_chunk(chunk_id=chunk_a, content="content a"),
            _vec_chunk(chunk_id=chunk_b, content="content b"),
        ]
        keyword_results = [_kw_chunk(chunk_id=chunk_a, content="content a")]

        result = reciprocal_rank_fusion(vector_results, keyword_results)

        assert len(result) == 2
        assert result[0].chunk_id == chunk_a
        assert result[0].vector_score is not None
        assert result[0].keyword_score is not None
        assert result[0].rrf_score > result[1].rrf_score

    def test_duplicates_merged_with_combined_scores(self) -> None:
        chunk_id = uuid4()
        vector_results = [_vec_chunk(chunk_id=chunk_id, content="c")]
        keyword_results = [_kw_chunk(chunk_id=chunk_id, content="c")]

        result = reciprocal_rank_fusion(vector_results, keyword_results)

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score is not None
        assert result[0].keyword_score is not None

    def test_custom_k_constant(self) -> None:
        chunk_id = uuid4()
        v = [_vec_chunk(chunk_id=chunk_id, content="c")]
        result_default = reciprocal_rank_fusion(v, [], k=60)
        result_custom = reciprocal_rank_fusion(v, [], k=100)

        assert result_default[0].rrf_score != result_custom[0].rrf_score
