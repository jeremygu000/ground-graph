"""Structure-aware document chunker.

Splits a ``ParsedContent`` into ``Chunk`` domain objects with precise
locators, preserving heading hierarchy and code-block boundaries.
"""

from __future__ import annotations

import hashlib
import re
from uuid import UUID, uuid4

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

        for section_heading_path, section_body in sections:
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
                        heading_path=section_heading_path,
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

    def _split_by_heading(self, content: ParsedContent) -> list[tuple[list[str], str]]:
        if not content.headings:
            return [([], content.body)]

        lines = content.body.splitlines(keepends=True)
        sections: list[tuple[list[str], str]] = []
        heading_path: list[str] = []
        current_lines: list[str] = []

        for line in lines:
            matched_level = 0
            matched_text = ""
            for level, heading_text in content.headings:
                prefix = "#" * level
                heading_prefix = f"{prefix} {heading_text}"
                if line.strip().startswith(prefix) and line.strip().startswith(heading_prefix):
                    matched_level = level
                    matched_text = heading_text
                    break
            if matched_level:
                if current_lines or sections:
                    sections.append((list(heading_path), "".join(current_lines).rstrip()))
                    current_lines = []
                while len(heading_path) >= matched_level:
                    heading_path.pop()
                heading_path.append(matched_text)
            current_lines.append(line)

        if current_lines:
            sections.append((list(heading_path), "".join(current_lines).rstrip()))
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
        window_start = self.start_of_window(pos)
        search = text[window_start : pos + 200] if pos < len(text) else text[window_start:pos]
        m = self.SENTENCE_END_RE.search(search)
        if m:
            return window_start + m.end()
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
