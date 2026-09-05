"""Unit tests for the structure-aware chunker."""

from __future__ import annotations

import re
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

    def test_locator_is_precise_line_range(self) -> None:
        content = ParsedContent(
            title="X", body="line one\nline two\nline three", media_type="text/plain"
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) == 1
        assert chunks[0].start_locator is not None
        assert chunks[0].end_locator is not None
        assert re.match(r"L\d+", chunks[0].start_locator)

    def test_end_locator_is_not_none(self) -> None:
        content = ParsedContent(title="X", body="hello world test content", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert chunks[0].end_locator is not None

    def test_code_block_preserved_not_split(self) -> None:
        body = "# Intro\n\nParagraph.\n\n```python\ndef foo():\n    pass\n```\n\n## Conclusion"
        content = ParsedContent(
            title="Code",
            body=body,
            media_type="text/markdown",
            headings=[(1, "Intro"), (2, "Conclusion")],
            code_blocks=[(5, 7)],
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        for chunk in chunks:
            if "def foo" in chunk.content:
                assert "```python" in chunk.content
                assert "```" in chunk.content

    def test_table_preserved_not_split(self) -> None:
        body = "# Data\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nEnd."
        content = ParsedContent(
            title="Table",
            body=body,
            media_type="text/markdown",
            headings=[(1, "Data")],
            tables=[(3, 5)],
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        for chunk in chunks:
            if "A" in chunk.content and "B" in chunk.content:
                assert "1" in chunk.content
                assert "2" in chunk.content

    def test_list_item_preserved_not_split(self) -> None:
        body = "# Steps\n\n- item one\n- item two\n\nDone."
        content = ParsedContent(
            title="List",
            body=body,
            media_type="text/markdown",
            headings=[(1, "Steps")],
            list_items=[(3, 4)],
        )
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        for chunk in chunks:
            if "item one" in chunk.content:
                assert "item two" in chunk.content

    def test_empty_body_returns_no_chunks(self) -> None:
        content = ParsedContent(title="X", body="", media_type="text/plain")
        chunks = self.chunker.chunk(content, self.document_id, self.version_id, self.principals)
        assert len(chunks) == 0
