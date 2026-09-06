"""Unit tests for RRF fusion."""

from __future__ import annotations

from uuid import uuid4

import pytest

from groundgraph.application.retrieval.fusion import reciprocal_rank_fusion
from groundgraph.infrastructure.postgres.keyword_retriever import KeywordSearchResult
from groundgraph.infrastructure.postgres.vector_retriever import VectorSearchResult


class TestReciprocalRankFusion:
    def test_empty_results_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_vector_only_results(self) -> None:
        chunk_id = uuid4()
        source_id = uuid4()
        doc_id = uuid4()
        ver_id = uuid4()

        vector_results = [
            VectorSearchResult(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="test content",
                score=0.1,
                allowed_principals=["eng"],
            )
        ]

        result = reciprocal_rank_fusion(vector_results, [])

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score == 0.1
        assert result[0].keyword_score is None
        assert result[0].rrf_score == pytest.approx(1 / 61)

    def test_keyword_only_results(self) -> None:
        chunk_id = uuid4()
        source_id = uuid4()
        doc_id = uuid4()
        ver_id = uuid4()

        keyword_results = [
            KeywordSearchResult(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="test content",
                rank=0.05,
                allowed_principals=["eng"],
            )
        ]

        result = reciprocal_rank_fusion([], keyword_results)

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score is None
        assert result[0].keyword_score == 0.05

    def test_fusion_combines_both_sources(self) -> None:
        chunk_a = uuid4()
        chunk_b = uuid4()
        source_id = uuid4()
        doc_id = uuid4()
        ver_id = uuid4()

        vector_results = [
            VectorSearchResult(
                chunk_id=chunk_a,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content a",
                score=0.1,
                allowed_principals=["eng"],
            ),
            VectorSearchResult(
                chunk_id=chunk_b,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content b",
                score=0.2,
                allowed_principals=["eng"],
            ),
        ]

        keyword_results = [
            KeywordSearchResult(
                chunk_id=chunk_a,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content a",
                rank=0.05,
                allowed_principals=["eng"],
            ),
        ]

        result = reciprocal_rank_fusion(vector_results, keyword_results)

        assert len(result) == 2
        assert result[0].chunk_id == chunk_a
        assert result[0].vector_score is not None
        assert result[0].keyword_score is not None
        assert result[0].rrf_score > result[1].rrf_score

    def test_duplicates_merged_with_combined_scores(self) -> None:
        chunk_id = uuid4()
        source_id = uuid4()
        doc_id = uuid4()
        ver_id = uuid4()

        vector_results = [
            VectorSearchResult(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content",
                score=0.1,
                allowed_principals=["eng"],
            ),
        ]

        keyword_results = [
            KeywordSearchResult(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content",
                rank=0.05,
                allowed_principals=["eng"],
            ),
        ]

        result = reciprocal_rank_fusion(vector_results, keyword_results)

        assert len(result) == 1
        assert result[0].chunk_id == chunk_id
        assert result[0].vector_score is not None
        assert result[0].keyword_score is not None

    def test_custom_k_constant(self) -> None:
        chunk_id = uuid4()
        source_id = uuid4()
        doc_id = uuid4()
        ver_id = uuid4()

        vector_results = [
            VectorSearchResult(
                chunk_id=chunk_id,
                source_id=source_id,
                document_id=doc_id,
                version_id=ver_id,
                content="content",
                score=0.1,
                allowed_principals=["eng"],
            ),
        ]

        result_default = reciprocal_rank_fusion(vector_results, [], k=60)
        result_custom = reciprocal_rank_fusion(vector_results, [], k=100)

        assert result_default[0].rrf_score != result_custom[0].rrf_score
