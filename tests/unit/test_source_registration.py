"""Unit tests for source registration service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from groundgraph.application.ingestion.source_registration import SourceRegistrationService
from groundgraph.domain.documents import SourceDescriptor


class _FakeDocumentRepository:
    def __init__(self) -> None:
        self.created_sources: list[SourceDescriptor] = []

    async def create_source(self, source: SourceDescriptor) -> SourceDescriptor:
        self.created_sources.append(source)
        return source


class TestSourceRegistrationService:
    def setup_method(self) -> None:
        self.fake_repo: Any = _FakeDocumentRepository()
        self.service = SourceRegistrationService(self.fake_repo)

    async def test_register_filesystem_source(self) -> None:
        await self.service.register_filesystem_source(
            uri="/mnt/docs",
            classification="internal",
            tenant_id="tenant-a",
            allowed_principals=["engineering"],
        )

        assert len(self.fake_repo.created_sources) == 1
        source = self.fake_repo.created_sources[0]
        assert source.source_type == "filesystem"
        assert source.uri == "/mnt/docs"
        assert source.classification == "internal"
        assert source.tenant_id == "tenant-a"
        assert source.allowed_principals == ["engineering"]

    async def test_scan_directory_finds_markdown_files(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.txt").write_text("Hello")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "doc3.md").write_text("# Doc 3")

        files = await self.service.scan_directory(str(tmp_path), extensions=["md"])

        assert len(files) == 2
        assert all(f.endswith(".md") for f in files)

    async def test_scan_directory_with_extension_filter(self, tmp_path: Path) -> None:
        (tmp_path / "doc.txt").write_text("text")
        (tmp_path / "doc.md").write_text("markdown")
        (tmp_path / "doc.py").write_text("python")

        files = await self.service.scan_directory(str(tmp_path), extensions=["txt"])

        assert len(files) == 1
        assert files[0].endswith(".txt")

    async def test_scan_directory_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError, match="Not a directory"):
            await self.service.scan_directory("/nonexistent/path/xyz")

    def test_calculate_checksum(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_bytes(b"hello world")

        checksum = self.service.calculate_checksum(str(file_path))

        assert len(checksum) == 64
        assert checksum == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
