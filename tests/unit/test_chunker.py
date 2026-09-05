"""Unit tests for the structure-aware chunker."""

from __future__ import annotations

from uuid import uuid4

from groundgraph.application.ingestion.chunker import Chunker
from groundgraph.application.ingestion.parsers import ParsedContent


class TestChunker:
    def setup_method(self) -> None:
        self.chunker = Chunker()
        self.document_id = uuid4()
        self.version_id = uuid4()
        self.principals = ["engineering"]

    def test_chunks_small_text_as_one(self) -> None:
        content = ParsedContent(
            title="Small",
            body="Hello world.",
            media_type="text/plain",
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."
        assert chunks[0].ordinal == 0

    def test_chunks_longer_text_into_multiple(self) -> None:
        content = ParsedContent(
            title="Long",
            body=" ".join(["word"] * 3000),
            media_type="text/plain",
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) >= 2
        assert all(c.document_id == self.document_id for c in chunks)
        assert all(c.version_id == self.version_id for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_heading_path_preserved_in_chunks(self) -> None:
        content = ParsedContent(
            title="Doc",
            body="# Section One\nContent under section one.\n## Subsection A\nMore content.",
            media_type="text/markdown",
            headings=[(1, "Section One"), (2, "Subsection A")],
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert all("Section One" in c.heading_path for c in chunks)

    def test_checksum_is_sha256_prefix(self) -> None:
        content = ParsedContent(title="X", body="Hello world", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) == 1
        assert len(chunks[0].checksum) == 16
        assert chunks[0].checksum.isalnum()

    def test_token_count_estimated(self) -> None:
        content = ParsedContent(
            title="X",
            body="one two three four five",
            media_type="text/plain",
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert chunks[0].token_count > 0

    def test_allowed_principals_set(self) -> None:
        content = ParsedContent(title="X", body="body", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, ["admin", "eng"])
        assert all(c.allowed_principals == ["admin", "eng"] for c in chunks)

    def test_locator_contains_ordinal(self) -> None:
        content = ParsedContent(title="X", body="hello world test content", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert f"[{chunks[0].ordinal}]" in (chunks[0].start_locator or "")

    def test_empty_body_returns_no_chunks(self) -> None:
        content = ParsedContent(title="X", body="", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) == 0
