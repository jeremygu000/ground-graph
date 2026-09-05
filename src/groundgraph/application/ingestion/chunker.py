"""Structure-aware document chunker.

Splits a ``ParsedContent`` into ``Chunk`` domain objects with precise
locators, preserving heading hierarchy and code-block boundaries.
"""

from __future__ import annotations

import hashlib
import re
from uuid import UUID, uuid4

from groundgraph.domain.defaults import empty_str_list
from groundgraph.domain.documents import Chunk

from .parsers import ParsedContent

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
TOKENS_PER_CHAR = 0.25


class Chunker:
    def chunk(
        self,
        content: ParsedContent,
        document_id: UUID,
        version_id: UUID,
        allowed_principals: list[str],
    ) -> list[Chunk]:
        sections = self._split_by_heading(content)
        chunks: list[Chunk] = []
        ordinal = 0

        for section_heading, section_body in sections:
            heading_path = empty_str_list()
            if section_heading:
                heading_path = [section_heading]
            elif content.headings:
                heading_path = [h[1] for h in content.headings if h[0] == 1]

            sub_chunks = self._chunk_text(
                section_body,
                max_tokens=MAX_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
            )
            for sub_body in sub_chunks:
                chunks.append(
                    Chunk(
                        chunk_id=uuid4(),
                        document_id=document_id,
                        version_id=version_id,
                        ordinal=ordinal,
                        heading_path=heading_path,
                        content=sub_body,
                        token_count=self._estimate_tokens(sub_body),
                        checksum=self._checksum(sub_body),
                        start_locator=self._locator(ordinal, sub_body),
                        end_locator=None,
                        allowed_principals=allowed_principals,
                    )
                )
                ordinal += 1

        return chunks

    def _split_by_heading(self, content: ParsedContent) -> list[tuple[str, str]]:
        if not content.headings:
            return [("", content.body)]

        lines = content.body.splitlines(keepends=True)
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in lines:
            is_heading = False
            for level, heading_text in content.headings:
                prefix = "#" * level
                heading_prefix = f"{prefix} {heading_text}"
                if line.strip().startswith(prefix) and line.strip().startswith(heading_prefix):
                    is_heading = True
                    break
            if is_heading:
                if current_lines or sections:
                    sections.append((current_heading, "".join(current_lines).rstrip()))
                    current_lines = []
                m = re.match(r"^#+\s+(.+)$", line.strip())
                current_heading = m.group(1) if m else ""
            current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "".join(current_lines).rstrip()))
        return sections

    def _chunk_text(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        if not text.strip():
            return []
        estimated_chars = int(max_tokens / TOKENS_PER_CHAR)
        overlap_chars = int(overlap_tokens / TOKENS_PER_CHAR)

        if self._estimate_tokens(text) <= max_tokens:
            return [text.strip()]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + estimated_chars, text_len)
            if end < text_len:
                boundary = self._find_sentence_boundary(text, end)
                if boundary > start + estimated_chars // 2:
                    end = boundary
                else:
                    end = min(start + estimated_chars, text_len)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= text_len:
                break
            start = end - overlap_chars
            start = max(start, 0)

        return [c for c in chunks if c]

    SENTENCE_END_RE = re.compile(r"[.!?]\s+(?=[A-Z])")

    def _find_sentence_boundary(self, text: str, pos: int) -> int:
        if pos < len(text):
            search = text[self.start_of_window(pos) : pos + 200]
        else:
            search = text[pos - 200 : pos]
        m = self.SENTENCE_END_RE.search(search)
        if m:
            return pos - len(search) + m.end()
        return pos

    def start_of_window(self, pos: int) -> int:
        return max(0, pos - 200)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text) * TOKENS_PER_CHAR)

    @staticmethod
    def _checksum(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _locator(ordinal: int, text: str) -> str:
        first = text[:40].replace("\n", " ").strip()
        return f"[{ordinal}] {first}..."
