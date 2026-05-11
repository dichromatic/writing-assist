"""
Shared parser helpers for span-based document parsing.

# Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from backend.nlp.types import (
    Heading,
    Paragraph,
    Scene,
    SceneBreak,
    Section,
    SectionAnchor,
)

_Span = Union[Heading, Paragraph, SceneBreak]


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
