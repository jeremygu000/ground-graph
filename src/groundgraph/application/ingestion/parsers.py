"""Document parsers for supported formats.

Each parser returns a structured ``ParsedContent`` that the chunker consumes.
PDF, DOCX, and EPUB are noted as deferred (M3 acceptance criteria covers
Markdown, text, and HTML as the primary formats).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


def _metadata_default() -> dict[str, object]:
    return {}


def _headings_default() -> list[tuple[int, str]]:
    return []


@dataclass
class ParsedContent:
    title: str
    body: str
    media_type: str
    metadata: dict[str, object] = field(default_factory=_metadata_default)
    headings: list[tuple[int, str]] = field(default_factory=_headings_default)


class BaseParser(ABC):
    @property
    @abstractmethod
    def media_type(self) -> str: ...

    @abstractmethod
    def parse(self, content: bytes) -> ParsedContent: ...


class TextParser(BaseParser):
    @property
    def media_type(self) -> str:
        return "text/plain"

    def parse(self, content: bytes) -> ParsedContent:
        text = content.decode("utf-8", errors="replace")
        first_line = text.split("\n", maxsplit=1)[0].strip()
        title = first_line[:120] if first_line else "untitled"
        return ParsedContent(
            title=title,
            body=text,
            media_type=self.media_type,
            metadata={"charset": "utf-8"},
        )


class MarkdownParser(BaseParser):
    @property
    def media_type(self) -> str:
        return "text/markdown"

    def parse(self, content: bytes) -> ParsedContent:
        text = content.decode("utf-8", errors="replace")
        title, body = self._strip_frontmatter(text)
        headings = self._extract_headings(body)
        return ParsedContent(
            title=title or "untitled",
            body=body,
            media_type=self.media_type,
            metadata={},
            headings=headings,
        )

    def _strip_frontmatter(self, text: str) -> tuple[str, str]:
        fm_match = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
        if not fm_match:
            return self._title_from_first_header(text), text
        rest = text[fm_match.end() :]
        title = self._title_from_first_header(rest)
        return title, rest

    def _title_from_first_header(self, text: str) -> str:
        m = re.match(r"^#\s+(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else ""

    def _extract_headings(self, body: str) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for m in re.finditer(r"^(#{1,6})\s+(.+)$", body, re.MULTILINE):
            level = len(m.group(1))
            results.append((level, m.group(2).strip()))
        return results


class HtmlParser(BaseParser):
    @property
    def media_type(self) -> str:
        return "text/html"

    def parse(self, content: bytes) -> ParsedContent:
        text = content.decode("utf-8", errors="replace")
        title = self._extract_title(text)
        body = self._strip_tags(text)
        headings = self._extract_headings(text)
        return ParsedContent(
            title=title or "untitled",
            body=body,
            media_type=self.media_type,
            metadata={},
            headings=headings,
        )

    def _extract_title(self, html: str) -> str:
        m = re.search(r"<title[^>]*>(.+?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"<h1[^>]*>(.+?)</h1>", html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    STRIP_TAGS_RE = re.compile(r"<[^>]+>")
    WHITESPACE_RE = re.compile(r"[ \t]+")

    def _strip_tags(self, html: str) -> str:
        text = self.STRIP_TAGS_RE.sub(" ", html)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = self.WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def _extract_headings(self, html: str) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for m in re.finditer(r"<h([1-6])[^>]*>(.+?)</h\1>", html, re.IGNORECASE | re.DOTALL):
            level = int(m.group(1))
            text = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            results.append((level, text))
        return results


class ParserRegistry:
    _parsers: ClassVar[dict[str, BaseParser]] = {}

    @classmethod
    def register(cls, parser: BaseParser) -> None:
        cls._parsers[parser.media_type] = parser

    @classmethod
    def get(cls, media_type: str) -> BaseParser | None:
        return cls._parsers.get(media_type)


for _parser_cls in (TextParser, MarkdownParser, HtmlParser):
    ParserRegistry.register(_parser_cls())
