"""Entity resolution service.

Resolves entity mentions to canonical entities using a combination of
string similarity and LLM-based disambiguation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from groundgraph.domain.knowledge import CanonicalEntity, EntityMention

if TYPE_CHECKING:
    from groundgraph.infrastructure.neo4j.repository import Neo4jGraphRepository

_MIN_CANDIDATE_SIMILARITY = 0.5


class EntityResolutionService:
    """Resolve entity mentions to canonical entities.

    Resolution strategy:
      1. Exact string match → direct resolve
      2. Case-insensitive match → resolve with high confidence
      3. Fuzzy match (Levenshtein) → candidate for review if confidence > threshold
      4. No match → create new canonical entity
    """

    def __init__(
        self,
        graph_repository: Neo4jGraphRepository | None = None,
        fuzzy_threshold: float = 0.85,
        review_threshold: float = 0.7,
        read_only: bool = False,
    ) -> None:
        self._graph_repository = graph_repository
        self._fuzzy_threshold = fuzzy_threshold
        self._review_threshold = review_threshold
        self._read_only = read_only

    async def resolve(self, mention: EntityMention) -> CanonicalEntity | None:
        """Resolve a single mention to a canonical entity.

        Returns None if the mention is ambiguous and needs human review.
        """
        surface = mention.surface_form.strip()
        if not surface:
            return None

        candidates = await self._find_candidates(surface, mention.candidate_type)
        if not candidates:
            return await self._create_new(surface, mention.candidate_type)

        best = max(candidates, key=lambda c: c[1])
        canonical, score = best

        if score >= self._fuzzy_threshold:
            return canonical

        if score >= self._review_threshold:
            return None

        return await self._create_new(surface, mention.candidate_type)

    async def _find_candidates(
        self, surface: str, candidate_type: str
    ) -> list[tuple[CanonicalEntity, float]]:
        if self._graph_repository is None:
            return []
        entities = await self._graph_repository.find_entities(entity_type=candidate_type)
        results: list[tuple[CanonicalEntity, float]] = []
        for entity in entities:
            sim = self._string_similarity(surface, entity.canonical_name)
            if sim >= _MIN_CANDIDATE_SIMILARITY:
                results.append((entity, sim))
            for alias in entity.aliases:
                sim = self._string_similarity(surface, alias)
                if sim >= _MIN_CANDIDATE_SIMILARITY:
                    results.append((entity, sim))
        return results

    def _string_similarity(self, a: str, b: str) -> float:
        a_lower = a.lower()
        b_lower = b.lower()
        if a_lower == b_lower:
            return 1.0
        if a_lower in b_lower or b_lower in a_lower:
            return 0.9
        len_max = max(len(a), len(b))
        if len_max == 0:
            return 0.0
        dist = self._levenshtein(a_lower, b_lower)
        return 1.0 - dist / len_max

    def _levenshtein(self, a: str, b: str) -> int:
        if len(a) < len(b):
            return self._levenshtein(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

    async def _create_new(self, name: str, entity_type: str) -> CanonicalEntity:
        entity = CanonicalEntity(
            entity_id=uuid4(),
            entity_type=entity_type,
            canonical_name=name,
            aliases=[name],
        )
        if self._graph_repository is not None and not self._read_only:
            await self._graph_repository.create_entity(entity)
        return entity
