"""Filesystem source registration and file acquisition."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import UUID

from groundgraph.application.ports import DocumentRepository
from groundgraph.domain.documents import SourceDescriptor


class SourceRegistrationService:
    def __init__(self, documents: DocumentRepository) -> None:
        self._documents = documents

    async def register_filesystem_source(
        self,
        uri: str,
        classification: str,
        tenant_id: str,
        allowed_principals: list[str],
    ) -> SourceDescriptor:
        source = SourceDescriptor(
            source_id=UUID(int=0),
            source_type="filesystem",
            uri=uri,
            classification=classification,
            tenant_id=tenant_id,
            allowed_principals=allowed_principals,
        )
        return await self._documents.create_source(source)

    async def scan_directory(
        self,
        root_uri: str,
        extensions: list[str] | None = None,
    ) -> list[str]:
        def _scan() -> list[str]:
            root = Path(root_uri)
            if not root.is_dir():
                raise ValueError(f"Not a directory: {root_uri}")
            patterns = [f"*.{ext}" for ext in (extensions or ["md", "txt", "html", "py", "ts"])]
            files: list[str] = []
            for pattern in patterns:
                files.extend(str(p) for p in root.rglob(pattern))
            return sorted(files)

        return await asyncio.to_thread(_scan)

    @staticmethod
    def calculate_checksum(file_path: str) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
