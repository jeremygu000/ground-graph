"""Structure-aware document chunker.

Splits a ``ParsedContent`` into ``Chunk`` domain objects with precise
line-range locators, preserving heading hierarchy, fenced code-block
boundaries, table boundaries, and list-item boundaries.
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

        for section_heading_path, section_body, section_start_line in sections:
            sub_chunks = self._chunk_text(
                section_body,
                section_start_line,
                max_tokens=MAX_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
                content=content,
            )
            for sub_body, sub_start, sub_end in sub_chunks:
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
                        start_locator=self._locate(sub_start, sub_body),
                        end_locator=self._locate(sub_end, sub_body),
                        allowed_principals=allowed_principals,
                    )
                )
                ordinal += 1

        return chunks

    def _split_by_heading(self, content: ParsedContent) -> list[tuple[list[str], str, int]]:
        if not content.headings:
            return [([], content.body, 1)]

        lines = content.body.splitlines(keepends=True)
        sections: list[tuple[list[str], str, int]] = []
        heading_path: list[str] = []
        current_lines: list[str] = []
        current_start = 1

        for line_idx, line in enumerate(lines, start=1):
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
                    sections.append(
                        (list(heading_path), "".join(current_lines).rstrip(), current_start)
                    )
                    current_lines = []
                while len(heading_path) >= matched_level:
                    heading_path.pop()
                heading_path.append(matched_text)
                current_start = line_idx
            current_lines.append(line)

        if current_lines:
            sections.append((list(heading_path), "".join(current_lines).rstrip(), current_start))
        return sections

    def _chunk_text(
        self,
        text: str,
        section_start: int,
        max_tokens: int,
        overlap_tokens: int,
        content: ParsedContent,
    ) -> list[tuple[str, int, int]]:
        if not text.strip():
            return []

        estimated_chars = int(max_tokens / TOKENS_PER_CHAR)
        overlap_chars = int(overlap_tokens / TOKENS_PER_CHAR)

        if self._estimate_tokens(text) <= max_tokens:
            return [(text.strip(), section_start, self._last_line_of(text, section_start))]

        protected_ranges = self._build_protected_ranges(content, section_start)

        chunks: list[tuple[str, int, int]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            chunk_end = min(start + estimated_chars, text_len)

            safe_end = self._find_safe_end(text, chunk_end, protected_ranges, start)
            end = safe_end if safe_end is not None else chunk_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_start_line = section_start + text[:start].count("\n")
                chunk_end_line = section_start + text[:end].count("\n")
                chunks.append((chunk_text, chunk_start_line, chunk_end_line))

            if end >= text_len:
                break
            start = max(end - overlap_chars, chunks[-1][1] if chunks else 0)
            start = max(start, 0)

        return chunks

    def _build_protected_ranges(
        self, content: ParsedContent, section_start: int
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for cb_start, cb_end in content.code_blocks:
            if self._ranges_overlap(cb_start, cb_end, section_start, section_start + 999999):
                ranges.append((cb_start, cb_end))
        for tb_start, tb_end in content.tables:
            if self._ranges_overlap(tb_start, tb_end, section_start, section_start + 999999):
                ranges.append((tb_start, tb_end))
        for li_start, li_end in content.list_items:
            if self._ranges_overlap(li_start, li_end, section_start, section_start + 999999):
                ranges.append((li_start, li_end))
        ranges.sort()
        return ranges

    def _ranges_overlap(self, a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start <= b_end and b_start <= a_end

    def _find_safe_end(
        self, text: str, target: int, protected_ranges: list[tuple[int, int]], chunk_start: int
    ) -> int | None:
        for p_start, p_end in protected_ranges:
            if p_start > chunk_start and p_start < target:
                return p_start
            if p_start <= chunk_start <= p_end and p_end > target:
                return None
            if p_start < target <= p_end:
                return p_start
        return None

    def _last_line_of(self, text: str, start_line: int) -> int:
        return start_line + text.count("\n")

    SENTENCE_END_RE = re.compile(r"[.!?]\s+(?=[A-Z])")

    def _find_sentence_boundary(self, text: str, pos: int) -> int:
        window_start = max(0, pos - 200)
        search = text[window_start:pos]
        m = self.SENTENCE_END_RE.search(search)
        if m:
            return window_start + m.end()
        return pos

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text) * TOKENS_PER_CHAR)

    @staticmethod
    def _checksum(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _locate(line: int, text: str) -> str:
        first_line = text.split("\n", 1)[0][:60].replace("\n", " ").strip()
        return f"L{line} {first_line}"
