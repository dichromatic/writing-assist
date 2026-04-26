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
from typing import Optional, Union

from backend.nlp.types import (
    Heading,
    Paragraph,
    ParsedMarkdownDocument,
    Scene,
    SceneBreak,
    Section,
    SectionAnchor,
    SpanAnchor,
    stable_hash_id,
)

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')
# CommonMark thematic break: three or more of ---, ***, or ___ on a line by
# themselves with optional surrounding whitespace. All three styles are treated
# as scene breaks so manuscripts that use asterisms (***) or underscores work
# without requiring the author to convert to dashes.
_SCENE_BREAK_RE = re.compile(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$')

_Span = Union[Heading, Paragraph, SceneBreak]


def _normalize_text(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip edges."""
    return ' '.join(text.split())


def _strip_closing_hashes(text: str) -> str:
    """Remove optional trailing ATX heading closing sequence (e.g. 'Title ##')."""
    return re.sub(r'\s+#+\s*$', '', text).strip()


def _derive_sections(
    path: str,
    all_spans: list[_Span],
    doc_end: int,
) -> list[Section]:
    """Group spans into sections bounded by headings.

    A new section begins whenever a Heading span is encountered. Content
    before the first heading (if any) forms a section with heading=None.
    An empty document produces one empty section.

    Args:
        path: Document path, used for SectionAnchor construction.
        all_spans: All span types merged and sorted by span_ordinal.
        doc_end: Character length of the source document.

    Returns:
        Sections in document order.
    """
    sections: list[Section] = []
    section_idx = 0
    current_heading: Optional[Heading] = None
    current_spans: list[_Span] = []
    section_start = 0

    def close_section() -> None:
        nonlocal section_idx
        if not current_spans:
            return
        sections.append(Section(
            section_index=section_idx,
            heading=current_heading,
            span_ordinals=[s.span_ordinal for s in current_spans],
            start_char=section_start,
            end_char=current_spans[-1].end_char,
            anchor=SectionAnchor(path=path, section_index=section_idx),
        ))
        section_idx += 1

    for span in all_spans:
        if isinstance(span, Heading):
            close_section()
            current_heading = span
            current_spans = [span]
            section_start = span.start_char
        else:
            current_spans.append(span)

    close_section()

    if not sections:
        sections.append(Section(
            section_index=0,
            heading=None,
            span_ordinals=[],
            start_char=0,
            end_char=doc_end,
            anchor=SectionAnchor(path=path, section_index=0),
        ))

    return sections


def _derive_scenes(
    all_spans: list[_Span],
    sections: list[Section],
    doc_end: int,
) -> list[Scene]:
    """Group spans into scenes bounded by SceneBreak markers.

    A new scene begins after each SceneBreak. The scene break itself is not
    included in either scene's span_ordinals - it is accessible through the
    ParsedMarkdownDocument.scene_breaks list.

    Args:
        all_spans: All span types merged and sorted by span_ordinal.
        sections: Derived sections, used to assign each scene a section_index.
        doc_end: Character length of the source document.

    Returns:
        Scenes in document order. Always at least one scene.
    """
    # Reverse lookup so each content span can be assigned to its section.
    ordinal_to_section: dict[int, int] = {}
    for section in sections:
        for ordinal in section.span_ordinals:
            ordinal_to_section[ordinal] = section.section_index

    def scene_section(spans: list[_Span]) -> int:
        if not spans:
            return 0
        return ordinal_to_section.get(spans[0].span_ordinal, 0)

    scenes: list[Scene] = []
    scene_idx = 0
    current_spans: list[_Span] = []
    scene_start = 0

    for span in all_spans:
        if isinstance(span, SceneBreak):
            scenes.append(Scene(
                scene_index=scene_idx,
                section_index=scene_section(current_spans),
                span_ordinals=[s.span_ordinal for s in current_spans],
                start_char=scene_start,
                end_char=span.start_char,
            ))
            scene_idx += 1
            current_spans = []
            scene_start = span.end_char
        else:
            current_spans.append(span)

    # Final scene after the last break (or the only scene if there are no breaks).
    scenes.append(Scene(
        scene_index=scene_idx,
        section_index=scene_section(current_spans),
        span_ordinals=[s.span_ordinal for s in current_spans],
        start_char=scene_start,
        end_char=current_spans[-1].end_char if current_spans else doc_end,
    ))

    return scenes


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
            normalized_text=_normalize_text(raw),
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
            heading_text = _strip_closing_hashes(m.group(2))
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
    sections = _derive_sections(path, all_spans, len(text))
    scenes = _derive_scenes(all_spans, sections, len(text))

    return ParsedMarkdownDocument(
        path=path,
        raw_text=text,
        headings=headings,
        paragraphs=paragraphs,
        scene_breaks=scene_breaks,
        sections=sections,
        scenes=scenes,
    )
