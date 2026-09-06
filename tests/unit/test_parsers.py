"""Unit tests for document parsers."""

from __future__ import annotations

from groundgraph.application.ingestion.parsers import (
    HtmlParser,
    MarkdownParser,
    ParserRegistry,
    TextParser,
)


class TestTextParser:
    def test_parses_plain_text(self) -> None:
        content = b"Hello world\nThis is a test document."
        result = TextParser().parse(content)

        assert result.title == "Hello world"
        assert result.body == "Hello world\nThis is a test document."
        assert result.media_type == "text/plain"
        assert result.metadata["charset"] == "utf-8"

    def test_title_truncated_at_120_chars(self) -> None:
        long_line = "x" * 150
        content = f"{long_line}\nbody".encode()
        result = TextParser().parse(content)
        assert len(result.title) == 120

    def test_missing_title_defaults_to_untitled(self) -> None:
        content = b""
        result = TextParser().parse(content)
        assert result.title == "untitled"


class TestMarkdownParser:
    def test_extracts_title_from_h1(self) -> None:
        content = b"# My Document Title\n\nSome body text."
        result = MarkdownParser().parse(content)

        assert result.title == "My Document Title"
        assert "Some body text" in result.body

    def test_strips_yaml_frontmatter(self) -> None:
        content = b"""---
title: Frontmatter Title
author: Test
---

# Actual Title

Body content.
"""
        result = MarkdownParser().parse(content)
        assert "Actual Title" in result.title
        assert "---" in result.body

    def test_extracts_headings(self) -> None:
        content = b"# H1\n## H2\n### H3\n\n## Another H2"
        result = MarkdownParser().parse(content)

        assert result.headings == [
            (1, "H1"),
            (2, "H2"),
            (3, "H3"),
            (2, "Another H2"),
        ]

    def test_handles_no_frontmatter_no_h1(self) -> None:
        content = b"Plain text without any markdown."
        result = MarkdownParser().parse(content)
        assert result.title == "untitled"
        assert "Plain text" in result.body


class TestHtmlParser:
    def test_extracts_title_from_title_tag(self) -> None:
        content = b"<html><head><title>Page Title</title></head><body><p>Hello</p></body></html>"
        result = HtmlParser().parse(content)
        assert result.title == "Page Title"

    def test_extracts_title_from_h1(self) -> None:
        content = b"<html><body><h1>Main Heading</h1><p>Content</p></body></html>"
        result = HtmlParser().parse(content)
        assert result.title == "Main Heading"

    def test_strips_tags_from_body(self) -> None:
        content = b"<html><body><p>Hello <strong>world</strong></p></body></html>"
        result = HtmlParser().parse(content)
        assert "<" not in result.body
        assert "Hello world" in result.body

    def test_extracts_headings(self) -> None:
        content = b"<html><body><h1>Level1</h1><h2>Level2</h2><h3>Level3</h3></body></html>"
        result = HtmlParser().parse(content)
        assert result.headings == [
            (1, "Level1"),
            (2, "Level2"),
            (3, "Level3"),
        ]


class TestParserRegistry:
    def test_returns_text_parser_for_text_plain(self) -> None:
        parser = ParserRegistry.get("text/plain")
        assert parser is not None
        assert isinstance(parser, TextParser)

    def test_returns_markdown_parser(self) -> None:
        parser = ParserRegistry.get("text/markdown")
        assert parser is not None
        assert isinstance(parser, MarkdownParser)

    def test_returns_html_parser(self) -> None:
        parser = ParserRegistry.get("text/html")
        assert parser is not None
        assert isinstance(parser, HtmlParser)

    def test_returns_none_for_unsupported(self) -> None:
        result = ParserRegistry.get("application/x-unsupported-format")
        assert result is None
