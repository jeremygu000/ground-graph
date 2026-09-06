"""Unit tests for document parsers."""

from __future__ import annotations

import io
import tempfile
from typing import Any, cast

import pytest
from docx import Document  # pyright: ignore[reportMissingTypeStubs]
from ebooklib import epub  # pyright: ignore[reportMissingTypeStubs]

from groundgraph.application.ingestion.parsers import (
    DocxParser,
    EpubParser,
    HtmlParser,
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    PyParser,
    TextParser,
    TsParser,
    UnsupportedFormatError,
    UnsupportedReason,
)


def _pdf_fixture() -> bytes:
    stream = b"BT /F1 18 Tf 72 720 Td (Golden PDF) Tj 0 -30 Td (PDF body) Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode())
        document.extend(obj)
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(document)


def _docx_fixture() -> bytes:
    document = Document()
    document.core_properties.title = "Golden DOCX"
    document.add_heading("Golden DOCX", level=1)
    document.add_paragraph("DOCX body")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _epub_fixture() -> bytes:
    book = epub.EpubBook()
    book.set_identifier("golden")
    book.set_title("Golden EPUB")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Golden EPUB", file_name="chapter.xhtml", lang="en")
    chapter.content = "<h1>Golden EPUB</h1><p>EPUB body</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    with tempfile.NamedTemporaryFile(suffix=".epub") as output:
        epub.write_epub(output.name, book)
        output.seek(0)
        return output.read()


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

    def test_empty_content_body_is_empty(self) -> None:
        content = b""
        result = TextParser().parse(content)
        assert result.body == ""

    def test_binary_content_decoded_with_replacement(self) -> None:
        content = b"\x00\xff\xfe invalid"
        result = TextParser().parse(content)
        assert result.title != "untitled"


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

    def test_empty_content(self) -> None:
        content = b""
        result = MarkdownParser().parse(content)
        assert result.title == "untitled"
        assert result.body == ""

    def test_only_frontmatter_no_trailing_newline(self) -> None:
        content = b"---\nkey: value\n---"
        result = MarkdownParser().parse(content)
        assert result.title == "untitled"


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

    def test_empty_html(self) -> None:
        content = b"<html><body></body></html>"
        result = HtmlParser().parse(content)
        assert result.title == "untitled"
        assert result.body == ""

    def test_malformed_html_recovered(self) -> None:
        content = b"<html><body><p>Unclosed tag<p>Another</body></html>"
        result = HtmlParser().parse(content)
        assert "Unclosed tag" in result.body


class TestPdfParser:
    def test_parses_minimal_valid_pdf(self) -> None:
        result = PdfParser().parse(_pdf_fixture())

        assert result.title == "Golden PDF"
        assert "PDF body" in result.body
        assert result.metadata["page_count"] == 1

    def test_empty_pdf_raises_parse_error(self) -> None:
        content = b""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            PdfParser().parse(content)
        assert exc_info.value.reason == UnsupportedReason.PARSE_ERROR


class TestDocxParser:
    def test_parses_minimal_valid_docx(self) -> None:
        result = DocxParser().parse(_docx_fixture())

        assert result.title == "Golden DOCX"
        assert "DOCX body" in result.body
        assert int(cast(Any, result.metadata["paragraph_count"])) >= 2

    def test_malformed_docx_raises(self) -> None:
        content = b"not a valid docx"
        with pytest.raises(UnsupportedFormatError) as exc_info:
            DocxParser().parse(content)
        assert exc_info.value.reason == UnsupportedReason.DOCX_PARSE_ERROR

    def test_empty_docx_raises(self) -> None:
        content = b""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            DocxParser().parse(content)
        assert exc_info.value.reason == UnsupportedReason.DOCX_PARSE_ERROR


class TestEpubParser:
    def test_parses_minimal_valid_epub(self) -> None:
        result = EpubParser().parse(_epub_fixture())

        assert result.title == "Golden EPUB"
        assert "Golden EPUB" in result.body
        assert "EPUB body" in result.body

    def test_empty_epub_raises(self) -> None:
        content = b""
        with pytest.raises(UnsupportedFormatError) as exc_info:
            EpubParser().parse(content)
        assert exc_info.value.reason == UnsupportedReason.EPUB_PARSE_ERROR


class TestPyParser:
    def test_parses_python_file(self) -> None:
        content = b"def hello():\n    print('world')\n"
        result = PyParser().parse(content)
        assert result.title == "def hello"
        assert result.body == "def hello():\n    print('world')\n"
        assert result.media_type == "text/x-python"
        assert (1, "def hello") in result.headings

    def test_extracts_class_definition(self) -> None:
        content = b"class MyClass:\n    def method(self):\n        pass\n"
        result = PyParser().parse(content)
        assert result.title == "class MyClass"
        assert (1, "class MyClass") in result.headings

    def test_parses_empty_python(self) -> None:
        content = b""
        result = PyParser().parse(content)
        assert result.title == "untitled"
        assert result.body == ""


class TestTsParser:
    def test_parses_typescript_file(self) -> None:
        content = b"function greet(name: string): void {\n  console.log(name);\n}\n"
        result = TsParser().parse(content)
        assert result.title == "function greet"
        assert result.body == "function greet(name: string): void {\n  console.log(name);\n}\n"
        assert result.media_type == "text/typescript"
        assert (1, "function greet") in result.headings

    def test_extracts_class_and_interface(self) -> None:
        content = b"interface Config {\n  key: string;\n}\nexport class Service {\n  run() {}\n}\n"
        result = TsParser().parse(content)
        assert result.title == "interface Config"
        headings_titles = [h[1] for h in result.headings]
        assert "interface Config" in headings_titles
        assert "export class Service" in headings_titles

    def test_parses_empty_typescript(self) -> None:
        content = b""
        result = TsParser().parse(content)
        assert result.title == "untitled"
        assert result.body == ""


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

    def test_get_with_reason_returns_parser_for_supported(self) -> None:
        parser = ParserRegistry.get_with_reason("text/plain")
        assert isinstance(parser, TextParser)

    def test_get_with_reason_raises_for_unsupported(self) -> None:
        with pytest.raises(UnsupportedFormatError) as exc_info:
            ParserRegistry.get_with_reason("application/x-unsupported")
        assert exc_info.value.reason == UnsupportedReason.UNSUPPORTED_MEDIA_TYPE
