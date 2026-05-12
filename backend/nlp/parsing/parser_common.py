"""
Shared parser helpers for span-based document parsing.

# Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Callable, Optional, Union

from backend.nlp.types import (
    Heading,
    ParsedMarkdownDocument,
    Paragraph,
    Scene,
    SceneBreak,
    Section,
    SectionAnchor,
    SpanAnchor,
)

_Span = Union[Heading, Paragraph, SceneBreak]


class LineClassification(Enum):
    """Line-level structural classification emitted by parser callbacks."""

    HEADING = "heading"
    SCENE_BREAK = "scene_break"
    PARAGRAPH = "paragraph"
    BLANK = "blank"


@dataclass(frozen=True)
class ClassifiedLine:
    """Structured line classification used by the shared scanner.

    Args:
        classification: Structural class for the source line.
        heading_level: Heading depth when classification is HEADING.
        heading_normalized_text: Normalized heading text.
    """

    classification: LineClassification
    heading_level: int = 0
    heading_normalized_text: str = ""


def normalize_text(text: str) -> str:
    """Collapse whitespace runs to a single space and trim the result.

    Args:
        text: Raw heading or paragraph text.

    Returns:
        A retrieval-friendly normalized surface.
    """
    return " ".join(text.split())


def strip_closing_hashes(text: str) -> str:
    """Remove optional trailing ATX closing hashes from heading content.

    Args:
        text: Heading content after the leading hash prefix.

    Returns:
        The heading text without any Markdown closing hashes.
    """
    return re.sub(r"\s+#+\s*$", "", text).strip()


def scan_document(
    path: str,
    text: str,
    classify_line: Callable[[str], ClassifiedLine],
) -> ParsedMarkdownDocument:
    """Parse document lines into shared span model via callback classification.

    Args:
        path: Source document path.
        text: Raw source text.
        classify_line: Parser-specific line classifier callback.

    Returns:
        ParsedMarkdownDocument with headings, paragraphs, scene breaks, sections,
        and scenes.
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
        classified = classify_line(stripped)

        if classified.classification == LineClassification.BLANK:
            flush_paragraph()
            continue

        if classified.classification == LineClassification.HEADING:
            flush_paragraph()
            if not classified.heading_normalized_text:
                if not paragraph_lines:
                    paragraph_start = line_start
                paragraph_lines.append(line)
                continue
            end_char = line_start + len(stripped)
            span_ordinal = next_ordinal()
            headings.append(
                Heading(
                    text=stripped,
                    level=classified.heading_level,
                    normalized_text=classified.heading_normalized_text,
                    span_ordinal=span_ordinal,
                    start_char=line_start,
                    end_char=end_char,
                    anchor=make_anchor(span_ordinal, line_start, end_char),
                )
            )
            continue

        if classified.classification == LineClassification.SCENE_BREAK:
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


def derive_sections(
    path: str,
    all_spans: list[_Span],
    doc_end: int,
) -> list[Section]:
    """Group spans into sections bounded by heading spans.

    Args:
        path: Document path used for section anchors.
        all_spans: All parsed spans in document order.
        doc_end: Character length of the raw document text.

    Returns:
        A non-empty ordered section list.
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
        sections.append(
            Section(
                section_index=section_idx,
                heading=current_heading,
                span_ordinals=[span.span_ordinal for span in current_spans],
                start_char=section_start,
                end_char=current_spans[-1].end_char,
                anchor=SectionAnchor(path=path, section_index=section_idx),
            )
        )
        section_idx += 1

    for span in all_spans:
        if isinstance(span, Heading):
            close_section()
            current_heading = span
            current_spans = [span]
            section_start = span.start_char
            continue
        current_spans.append(span)

    close_section()

    if not sections:
        sections.append(
            Section(
                section_index=0,
                heading=None,
                span_ordinals=[],
                start_char=0,
                end_char=doc_end,
                anchor=SectionAnchor(path=path, section_index=0),
            )
        )

    return sections


def derive_scenes(
    all_spans: list[_Span],
    sections: list[Section],
    doc_end: int,
) -> list[Scene]:
    """Group spans into scenes bounded by scene-break markers.

    Args:
        all_spans: All parsed spans in document order.
        sections: Derived sections used to assign each scene a section index.
        doc_end: Character length of the raw document text.

    Returns:
        A non-empty ordered scene list.
    """
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
            scenes.append(
                Scene(
                    scene_index=scene_idx,
                    section_index=scene_section(current_spans),
                    span_ordinals=[item.span_ordinal for item in current_spans],
                    start_char=scene_start,
                    end_char=span.start_char,
                )
            )
            scene_idx += 1
            current_spans = []
            scene_start = span.end_char
            continue
        current_spans.append(span)

    scenes.append(
        Scene(
            scene_index=scene_idx,
            section_index=scene_section(current_spans),
            span_ordinals=[item.span_ordinal for item in current_spans],
            start_char=scene_start,
            end_char=current_spans[-1].end_char if current_spans else doc_end,
        )
    )

    return scenes
