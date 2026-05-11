"""
Markdown parser - converts raw Markdown text into a structured span model.

Handles ATX headings, prose paragraphs, and explicit scene-break markers.
Sections are derived from heading boundaries; scenes are derived from
scene-break boundaries. All spans carry character offsets into the original
text so downstream stages can construct source anchors without re-scanning.

.. code-block:: mermaid

    flowchart TD
        A[Raw Markdown text] --> B[Line scanner]
        B --> C{Line type}
        C -->|ATX heading| D[Heading span]
        C -->|Scene break marker| E[SceneBreak span]
        C -->|Blank line| F[Flush paragraph buffer]
        C -->|Content| G[Paragraph buffer]
        G -->|Next blank or structural line| F
        F --> H[Paragraph span]
        D & E & H --> I[All spans sorted by ordinal]
        I --> J[Derive sections from heading boundaries]
        I --> K[Derive scenes from scene-break boundaries]
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

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')
# CommonMark thematic break: three or more of ---, ***, or ___ on a line by
# themselves with optional surrounding whitespace. All three styles are treated
# as scene breaks so manuscripts that use asterisms (***) or underscores work
# without requiring the author to convert to dashes.
_SCENE_BREAK_RE = re.compile(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$')

_Span = Union[Heading, Paragraph, SceneBreak]


def parse(path: str, text: str) -> ParsedMarkdownDocument:
    """Parse a Markdown document into a structured span model.

    Emits Heading, Paragraph, and SceneBreak spans with character offsets,
    then derives Sections from heading boundaries and Scenes from scene-break
    boundaries. Offsets are Unicode code-point positions into the raw text,
    consistent with Python string indexing.

    Args:
        path: Document path. Used as the basis for all anchors in the output.
        text: Raw Markdown text. Not modified.

    Returns:
        ParsedMarkdownDocument with all spans, sections, and scenes.
    """
    headings: list[Heading] = []
    paragraphs: list[Paragraph] = []
    scene_breaks: list[SceneBreak] = []
    ordinal = 0

    para_lines: list[str] = []
    para_start = 0

    def next_ordinal() -> int:
        nonlocal ordinal
        n = ordinal
        ordinal += 1
        return n

    def make_anchor(ord_: int, start: int, end: int) -> SpanAnchor:
        return SpanAnchor(path=path, span_ordinal=ord_, start_char=start, end_char=end)

    def flush_paragraph() -> None:
        if not para_lines:
            return
        raw = ''.join(para_lines).rstrip()
        if not raw:
            para_lines.clear()
            return
        end = para_start + len(raw)
        ord_ = next_ordinal()
        paragraphs.append(Paragraph(
            text=raw,
            normalized_text=normalize_text(raw),
            span_ordinal=ord_,
            start_char=para_start,
            end_char=end,
            anchor=make_anchor(ord_, para_start, end),
        ))
        para_lines.clear()

    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        stripped = line.rstrip('\n\r')

        if not stripped or stripped.isspace():
            flush_paragraph()
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            flush_paragraph()
            level = len(m.group(1))
            heading_text = strip_closing_hashes(m.group(2))
            if not heading_text:
                # A heading with no content (e.g. "# ") carries no structural
                # information; treat it as a paragraph line rather than
                # creating an anonymous heading that could pollute section structure.
                if not para_lines:
                    para_start = line_start
                para_lines.append(line)
                continue
            end = line_start + len(stripped)
            ord_ = next_ordinal()
            headings.append(Heading(
                text=stripped,
                level=level,
                normalized_text=heading_text,
                span_ordinal=ord_,
                start_char=line_start,
                end_char=end,
                anchor=make_anchor(ord_, line_start, end),
            ))
            continue

        if _SCENE_BREAK_RE.match(stripped):
            flush_paragraph()
            end = line_start + len(stripped)
            ord_ = next_ordinal()
            scene_breaks.append(SceneBreak(
                span_ordinal=ord_,
                start_char=line_start,
                end_char=end,
                anchor=make_anchor(ord_, line_start, end),
            ))
            continue

        if not para_lines:
            para_start = line_start
        para_lines.append(line)

    flush_paragraph()

    all_spans: list[_Span] = sorted(
        [*headings, *paragraphs, *scene_breaks],
        key=lambda s: s.span_ordinal,
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
