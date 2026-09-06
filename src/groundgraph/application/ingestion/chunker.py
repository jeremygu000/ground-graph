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
                        end_locator=self._locate_end(sub_end, sub_body),
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

        text_len = len(text)
        line_to_char = self._build_line_to_char_map(text)
        protected = self._build_protected_ranges_char(
            text, content, section_start, line_to_char, text_len
        )

        chunks: list[tuple[str, int, int]] = []
        start = 0
        prev_end: int | None = None

        while start < text_len:
            target = min(start + estimated_chars, text_len)

            safe_end = self._find_safe_end(text, target, protected, start, prev_end)

            end = safe_end if safe_end is not None else target

            if end <= start:
                end = min(start + 1, text_len)

            chunk_text = text[start:end].strip()
            if not chunk_text:
                start = min(start + 1, text_len)
                continue

            chunk_start_line = section_start + text[:start].count("\n")
            chunk_end_line = chunk_start_line + text[start:end].count("\n")
            chunks.append((chunk_text, chunk_start_line, chunk_end_line))

            prev_end = end
            if end >= text_len:
                break

            next_start = end - overlap_chars
            if len(chunks) > 1:
                prev_start_char = line_to_char.get(chunks[-2][2] - section_start + 1, 0)
                next_start = max(next_start, prev_start_char + 1)

            next_start = max(next_start, start + 1)

            for p_start, p_end in protected:
                if p_start < next_start < p_end:
                    next_start = p_end
                    break

            start = min(next_start, text_len)

        return chunks

    def _build_line_to_char_map(self, text: str) -> dict[int, int]:
        line_to_char: dict[int, int] = {}
        char_pos = 0
        for line_idx, line in enumerate(text.splitlines(), start=1):
            line_to_char[line_idx] = char_pos
            char_pos += len(line) + 1
        return line_to_char

    def _build_protected_ranges_char(
        self,
        text: str,
        content: ParsedContent,
        section_start: int,
        line_to_char: dict[int, int],
        text_length: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        section_rel_start = 1
        section_rel_end = section_rel_start + text.count("\n")

        def add_ranges(items: list[tuple[int, int]]) -> None:
            for item_start, item_end in items:
                if item_end < section_start or item_start > section_rel_end + section_start - 1:
                    continue
                rel_start = item_start - section_start + 1
                rel_end = item_end - section_start + 1
                if rel_start <= 0 or rel_end <= 0:
                    continue
                char_start = line_to_char.get(rel_start, 0)
                char_end = line_to_char.get(rel_end + 1, text_length)
                if char_start < char_end:
                    ranges.append((char_start, char_end))

        add_ranges(content.code_blocks)
        add_ranges(content.tables)
        add_ranges(content.list_items)
        ranges.sort()
        return ranges

    def _ranges_overlap(self, a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start <= b_end and b_start <= a_end

    def _find_safe_end(
        self,
        text: str,
        target: int,
        protected: list[tuple[int, int]],
        chunk_start: int,
        prev_end: int | None = None,
    ) -> int | None:
        for p_start, p_end in protected:
            if p_start > chunk_start and p_start < target:
                return p_end if p_start == prev_end else p_start
            if p_start <= chunk_start <= p_end and p_end > target:
                return p_end
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

    @staticmethod
    def _locate_end(line: int, text: str) -> str:
        lines = text.split("\n")
        last_line = (lines[-1] if lines else "").strip()
        last_line_preview = last_line[:60].replace("\n", " ").strip()
        return f"L{line} {last_line_preview}"
