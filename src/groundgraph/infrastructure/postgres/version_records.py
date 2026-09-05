"""Version records repository (plan.md §5.1 prompt/model/index version tables).

Provides CRUD operations for prompt_versions, model_config_versions, and
index_versions tables.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from groundgraph.infrastructure.postgres.models import (
    IndexVersion as IndexVersionModel,
)
from groundgraph.infrastructure.postgres.models import (
    ModelConfigVersion as ModelConfigVersionModel,
)
from groundgraph.infrastructure.postgres.models import (
    PromptVersion as PromptVersionModel,
)


class PromptVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        bundle_id: str,
        version: str,
        prompt_template: dict[str, Any],
        model_name: str,
    ) -> UUID:
        model = PromptVersionModel(
            bundle_id=bundle_id,
            version=version,
            prompt_template=prompt_template,
            model_name=model_name,
        )
        self._session.add(model)
        await self._session.flush()
        return model.version_id

    async def get(self, version_id: UUID) -> PromptVersionModel | None:
        result = await self._session.execute(
            select(PromptVersionModel).where(PromptVersionModel.version_id == version_id)
        )
        return result.scalar_one_or_none()


class ModelConfigVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, config_key: str, version: str, config: dict[str, Any]) -> UUID:
        model = ModelConfigVersionModel(
            config_key=config_key,
            version=version,
            config=config,
        )
        self._session.add(model)
        await self._session.flush()
        return model.version_id

    async def get(self, version_id: UUID) -> ModelConfigVersionModel | None:
        result = await self._session.execute(
            select(ModelConfigVersionModel).where(ModelConfigVersionModel.version_id == version_id)
        )
        return result.scalar_one_or_none()


class IndexVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        index_name: str,
        version: str,
        embedding_model: str,
        embedding_dimensions: int,
        chunker_version: str,
    ) -> UUID:
        model = IndexVersionModel(
            index_name=index_name,
            version=version,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            chunker_version=chunker_version,
        )
        self._session.add(model)
        await self._session.flush()
        return model.version_id

    async def get(self, version_id: UUID) -> IndexVersionModel | None:
        result = await self._session.execute(
            select(IndexVersionModel).where(IndexVersionModel.version_id == version_id)
        )
        return result.scalar_one_or_none()

    async def deactivate_previous(self, index_name: str, keep_version: UUID) -> None:
        await self._session.execute(
            update(IndexVersionModel)
            .where(IndexVersionModel.index_name == index_name)
            .where(IndexVersionModel.version_id != keep_version)
            .values(is_active=False)
        )
