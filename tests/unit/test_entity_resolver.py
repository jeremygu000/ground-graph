"""Unit tests for entity resolution service."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from groundgraph.application.extraction.entity_resolver import (
    EntityResolutionService,
)
from groundgraph.domain.knowledge import CanonicalEntity, EntityMention


def _make_mention(surface_form: str, entity_type: str = "Service") -> EntityMention:
    return EntityMention(
        mention_id=uuid4(),
        chunk_id=uuid4(),
        surface_form=surface_form,
        candidate_type=entity_type,
        extraction_confidence=0.9,
    )


def _make_canonical(
    name: str,
    entity_type: str = "Service",
    aliases: list[str] | None = None,
) -> CanonicalEntity:
    return CanonicalEntity(
        entity_id=uuid4(),
        entity_type=entity_type,
        canonical_name=name,
        aliases=aliases or [name],
    )


class TestStringSimilarity:
    """Tests for _string_similarity method."""

    def setup_method(self) -> None:
        self.svc = EntityResolutionService()

    def test_exact_match_returns_1(self) -> None:
        """Exact string match returns 1.0."""
        assert self.svc._string_similarity("AuthService", "AuthService") == 1.0

    def test_case_insensitive_match_returns_1(self) -> None:
        """Case-insensitive exact match returns 1.0."""
        assert self.svc._string_similarity("authservice", "AuthService") == 1.0

    def test_substring_match_returns_09(self) -> None:
        """Substring match returns 0.9."""
        assert self.svc._string_similarity("Auth", "AuthService") == 0.9
        assert self.svc._string_similarity("AuthService", "Auth") == 0.9

    def test_levenshtein_similarity(self) -> None:
        """Levenshtein-based similarity works for similar strings."""
        sim = self.svc._string_similarity("AuthService", "AuthServce")
        assert 0.8 <= sim < 1.0

    def test_empty_strings_return_1(self) -> None:
        """Empty strings match exactly (both are empty)."""
        assert self.svc._string_similarity("", "") == 1.0

    def test_different_strings_low_similarity(self) -> None:
        """Completely different strings have low similarity."""
        sim = self.svc._string_similarity("AuthService", "RedisCache")
        assert sim < 0.5


class TestLevenshtein:
    """Tests for _levenshtein method."""

    def setup_method(self) -> None:
        self.svc = EntityResolutionService()

    def test_empty_b_returns_len_a(self) -> None:
        """Empty b returns len(a)."""
        assert self.svc._levenshtein("abc", "") == 3

    def test_empty_a_returns_len_b(self) -> None:
        """Empty a returns len(b)."""
        assert self.svc._levenshtein("", "abc") == 3

    def test_equal_strings_returns_0(self) -> None:
        """Equal strings return 0."""
        assert self.svc._levenshtein("abc", "abc") == 0

    def test_single_char_diff_returns_1(self) -> None:
        """Single character difference returns 1."""
        assert self.svc._levenshtein("abc", "abd") == 1

    def test_insertion_returns_1(self) -> None:
        """Character insertion returns 1."""
        assert self.svc._levenshtein("abc", "abxc") == 1

    def test_deletion_returns_1(self) -> None:
        """Character deletion returns 1."""
        assert self.svc._levenshtein("abxc", "abc") == 1


class TestEntityResolutionService:
    """Tests for EntityResolutionService.resolve method."""

    @pytest.mark.asyncio
    async def test_resolve_empty_surface_form_returns_none(self) -> None:
        """resolve() returns None for empty surface form."""
        svc = EntityResolutionService()
        mention = _make_mention("")
        result = await svc.resolve(mention)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_whitespace_surface_form_returns_none(self) -> None:
        """resolve() returns None for whitespace-only surface form."""
        svc = EntityResolutionService()
        mention = _make_mention("   ")
        result = await svc.resolve(mention)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_no_candidates_creates_new(self) -> None:
        """resolve() creates new canonical entity when no candidates found."""
        mock_repo = AsyncMock()
        mock_repo.find_entities = AsyncMock(return_value=[])
        svc = EntityResolutionService(graph_repository=mock_repo)

        mention = _make_mention("NewService")
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "NewService"
        mock_repo.create_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_no_graph_repository_creates_new(self) -> None:
        """resolve() creates new entity when no graph repository configured."""
        svc = EntityResolutionService(graph_repository=None)

        mention = _make_mention("NewService")
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "NewService"

    @pytest.mark.asyncio
    async def test_resolve_high_similarity_returns_canonical(self) -> None:
        """resolve() returns canonical when best candidate has high similarity."""
        mock_repo = AsyncMock()
        existing = _make_canonical("AuthService")
        mock_repo.find_entities = AsyncMock(return_value=[existing])
        svc = EntityResolutionService(graph_repository=mock_repo, fuzzy_threshold=0.85)

        mention = _make_mention("AuthService")  # Exact match
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "AuthService"

    @pytest.mark.asyncio
    async def test_resolve_medium_similarity_returns_none_for_review(self) -> None:
        """resolve() returns None when similarity is between review and fuzzy threshold.

        Uses a mock that returns a specific score to test the review threshold logic.
        """
        mock_repo = AsyncMock()
        svc = EntityResolutionService(
            graph_repository=mock_repo,
            fuzzy_threshold=0.9,
            review_threshold=0.7,
        )

        mention = _make_mention("SomeOtherService")

        mock_repo.find_entities = AsyncMock(return_value=[])

        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "SomeOtherService"

    @pytest.mark.asyncio
    async def test_resolve_low_similarity_creates_new(self) -> None:
        """resolve() creates new entity when similarity is below review threshold."""
        mock_repo = AsyncMock()
        existing = _make_canonical("Redis")
        mock_repo.find_entities = AsyncMock(return_value=[existing])
        svc = EntityResolutionService(
            graph_repository=mock_repo,
            fuzzy_threshold=0.9,
            review_threshold=0.7,
        )

        mention = _make_mention("CompletelyDifferentService")
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "CompletelyDifferentService"

    @pytest.mark.asyncio
    async def test_resolve_uses_alias_similarity(self) -> None:
        """resolve() considers entity aliases when finding candidates."""
        mock_repo = AsyncMock()
        existing = _make_canonical("AuthenticationService", aliases=["AuthService", "AuthSvc"])
        mock_repo.find_entities = AsyncMock(return_value=[existing])
        svc = EntityResolutionService(graph_repository=mock_repo, fuzzy_threshold=0.85)

        mention = _make_mention("AuthSvc")  # Matches alias
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "AuthenticationService"

    @pytest.mark.asyncio
    async def test_resolve_below_min_candidate_similarity_excluded(self) -> None:
        """Entities below MIN_CANDIDATE_SIMILARITY are excluded from candidates."""
        mock_repo = AsyncMock()
        very_different = _make_canonical("XYZCompletelyDifferent")
        mock_repo.find_entities = AsyncMock(return_value=[very_different])
        svc = EntityResolutionService(graph_repository=mock_repo)

        mention = _make_mention("AuthService")
        result = await svc.resolve(mention)

        assert result is not None
        assert result.canonical_name == "AuthService"
