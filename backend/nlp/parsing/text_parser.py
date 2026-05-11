"""
Plain-text parser - converts loose .txt notes into the shared span model.

The text corpus is structurally messy, so this parser keeps the same output
contract as the Markdown parser while using a small set of heading heuristics:
Markdown-style headings when present, uppercase title lines, and numbered
outline headings such as "0.1 - Mission Briefing".

.. code-block:: mermaid

    flowchart TD
        A[Raw text document] --> B[Line scanner]
        B --> C{Structural heuristic}
        C -->|Markdown heading| D[Heading span]
        C -->|Uppercase title line| D
        C -->|Numbered outline line| D
        C -->|Scene break marker| E[SceneBreak span]
        C -->|Blank line| F[Flush paragraph buffer]
        C -->|Other content| G[Paragraph buffer]
        G -->|Next blank or structural line| F
        F --> H[Paragraph span]
        D & E & H --> I[All spans sorted by ordinal]
        I --> J[Derive sections]
        I --> K[Derive scenes]
        J & K --> L[ParsedMarkdownDocument]
"""

from __future__ import annotations

import re
from typing import Union

from backend.nlp.parsing.parser_common import (
    derive_scenes,
    derive_sections,
    normalize_text,
    strip_closing_hashes,
)
from backend.nlp.types import (
    Heading,
    Paragraph,
    ParsedMarkdownDocument,
    SceneBreak,
    SpanAnchor,
)

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_SCENE_BREAK_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[—–-]\s+(.+?)\s*$")
_ALLOWED_HEADING_PUNCTUATION = frozenset({"-", "—", "–", ":", "/", "&", "'", '"', "“", "”", "(", ")"})

_Span = Union[Heading, Paragraph, SceneBreak]


def _looks_like_uppercase_heading(line: str) -> bool:
    """Return True when a line is a likely uppercase title heading.

    Args:
        line: A single source line with trailing newline already stripped.

    Returns:
        True when the line should become a heading span.
    """
    stripped = line.strip()
    if not stripped or ":" in stripped:
        return False

    alpha_chars = [char for char in stripped if char.isalpha()]
    if len(alpha_chars) < 4:
        return False

    uppercase_count = sum(1 for char in alpha_chars if char.isupper())
    if uppercase_count / len(alpha_chars) < 0.8:
        return False

    if len(stripped) > 120:
        return False

    for char in stripped:
        if char.isalnum() or char.isspace():
            continue
        if char not in _ALLOWED_HEADING_PUNCTUATION:
            return False

    return True


def _outline_heading_level(numbering: str) -> int:
    """Map a numbered outline prefix to a stable heading depth.

    Args:
        numbering: The matched prefix, for example ``0.1`` or ``3.2.4``.

    Returns:
        A heading level capped to the Markdown heading range 1-6.
    """
    return min(len(numbering.split(".")) + 1, 6)


def parse(path: str, text: str) -> ParsedMarkdownDocument:
    """Parse a plain-text document into the shared span model.

    Args:
        path: Document path used for anchor construction.
        text: Raw .txt document text.

    Returns:
        Parsed document with headings, paragraphs, sections, and scenes.
    """
    headings: list[Heading] = []
    paragraphs: list[Paragraph] = []
    scene_breaks: list[SceneBreak] = []
    ordinal = 0

    paragraph_lines: list[str] = []
    paragraph_start = 0

    def next_ordinal() -> int:
        nonlocal ordinal
        current = ordinal
        ordinal += 1
        return current

    def make_anchor(span_ordinal: int, start_char: int, end_char: int) -> SpanAnchor:
        return SpanAnchor(
            path=path,
            span_ordinal=span_ordinal,
            start_char=start_char,
            end_char=end_char,
        )

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        raw = "".join(paragraph_lines).rstrip()
        if not raw:
            paragraph_lines.clear()
            return
        end_char = paragraph_start + len(raw)
        span_ordinal = next_ordinal()
        paragraphs.append(
            Paragraph(
                text=raw,
                normalized_text=normalize_text(raw),
                span_ordinal=span_ordinal,
                start_char=paragraph_start,
                end_char=end_char,
                anchor=make_anchor(span_ordinal, paragraph_start, end_char),
            )
        )
        paragraph_lines.clear()

    position = 0
    for line in text.splitlines(keepends=True):
        line_start = position
        position += len(line)
        stripped = line.rstrip("\n\r")

        if not stripped or stripped.isspace():
            flush_paragraph()
            continue

        markdown_heading = _MARKDOWN_HEADING_RE.match(stripped)
        if markdown_heading:
            flush_paragraph()
            level = len(markdown_heading.group(1))
            heading_text = strip_closing_hashes(markdown_heading.group(2))
            if not heading_text:
                if not paragraph_lines:
                    paragraph_start = line_start
                paragraph_lines.append(line)
                continue
            end_char = line_start + len(stripped)
            span_ordinal = next_ordinal()
            headings.append(
                Heading(
                    text=stripped,
                    level=level,
                    normalized_text=heading_text,
                    span_ordinal=span_ordinal,
                    start_char=line_start,
                    end_char=end_char,
                    anchor=make_anchor(span_ordinal, line_start, end_char),
                )
            )
            continue

        if _SCENE_BREAK_RE.match(stripped):
            flush_paragraph()
            end_char = line_start + len(stripped)
            span_ordinal = next_ordinal()
            scene_breaks.append(
                SceneBreak(
                    span_ordinal=span_ordinal,
                    start_char=line_start,
                    end_char=end_char,
                    anchor=make_anchor(span_ordinal, line_start, end_char),
                )
            )
            continue

        outline_heading = _NUMBERED_HEADING_RE.match(stripped)
        if outline_heading:
            flush_paragraph()
            end_char = line_start + len(stripped)
            span_ordinal = next_ordinal()
            headings.append(
                Heading(
                    text=stripped,
                    level=_outline_heading_level(outline_heading.group(1)),
                    normalized_text=stripped.strip(),
                    span_ordinal=span_ordinal,
                    start_char=line_start,
                    end_char=end_char,
                    anchor=make_anchor(span_ordinal, line_start, end_char),
                )
            )
            continue

        if _looks_like_uppercase_heading(stripped):
            flush_paragraph()
            end_char = line_start + len(stripped)
            span_ordinal = next_ordinal()
            headings.append(
                Heading(
                    text=stripped,
                    level=1,
                    normalized_text=normalize_text(stripped),
                    span_ordinal=span_ordinal,
                    start_char=line_start,
                    end_char=end_char,
                    anchor=make_anchor(span_ordinal, line_start, end_char),
                )
            )
            continue

        if not paragraph_lines:
            paragraph_start = line_start
        paragraph_lines.append(line)

    flush_paragraph()

    all_spans: list[_Span] = sorted(
        [*headings, *paragraphs, *scene_breaks],
        key=lambda span: span.span_ordinal,
    )
    sections = derive_sections(path, all_spans, len(text))
    scenes = derive_scenes(all_spans, sections, len(text))

    return ParsedMarkdownDocument(
        path=path,
        raw_text=text,
        headings=headings,
        paragraphs=paragraphs,
        scene_breaks=scene_breaks,
        sections=sections,
        scenes=scenes,
    )
