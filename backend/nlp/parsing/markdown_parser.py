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

from backend.nlp.parsing.parser_common import (
    ClassifiedLine,
    LineClassification,
    scan_document,
    strip_closing_hashes,
)
from backend.nlp.types import ParsedMarkdownDocument

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')
# CommonMark thematic break: three or more of ---, ***, or ___ on a line by
# themselves with optional surrounding whitespace. All three styles are treated
# as scene breaks so manuscripts that use asterisms (***) or underscores work
# without requiring the author to convert to dashes.
_SCENE_BREAK_RE = re.compile(r'^\s*(-{3,}|\*{3,}|_{3,})\s*$')

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
    def classify_line(stripped: str) -> ClassifiedLine:
        if not stripped or stripped.isspace():
            return ClassifiedLine(classification=LineClassification.BLANK)
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            heading_text = strip_closing_hashes(m.group(2))
            return ClassifiedLine(
                classification=LineClassification.HEADING,
                heading_level=level,
                heading_normalized_text=heading_text,
            )
        if _SCENE_BREAK_RE.match(stripped):
            return ClassifiedLine(classification=LineClassification.SCENE_BREAK)
        return ClassifiedLine(classification=LineClassification.PARAGRAPH)

    return scan_document(path, text, classify_line)
