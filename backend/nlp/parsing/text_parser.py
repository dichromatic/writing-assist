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

from backend.nlp.parsing.parser_common import (
    ClassifiedLine,
    LineClassification,
    normalize_text,
    scan_document,
    strip_closing_hashes,
)
from backend.nlp.types import ParsedMarkdownDocument

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_SCENE_BREAK_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[—–-]\s+(.+?)\s*$")
_ALLOWED_HEADING_PUNCTUATION = frozenset({"-", "—", "–", ":", "/", "&", "'", '"', "“", "”", "(", ")"})

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
    def classify_line(stripped: str) -> ClassifiedLine:
        if not stripped or stripped.isspace():
            return ClassifiedLine(classification=LineClassification.BLANK)
        markdown_heading = _MARKDOWN_HEADING_RE.match(stripped)
        if markdown_heading:
            level = len(markdown_heading.group(1))
            heading_text = strip_closing_hashes(markdown_heading.group(2))
            return ClassifiedLine(
                classification=LineClassification.HEADING,
                heading_level=level,
                heading_normalized_text=heading_text,
            )

        if _SCENE_BREAK_RE.match(stripped):
            return ClassifiedLine(classification=LineClassification.SCENE_BREAK)

        outline_heading = _NUMBERED_HEADING_RE.match(stripped)
        if outline_heading:
            return ClassifiedLine(
                classification=LineClassification.HEADING,
                heading_level=_outline_heading_level(outline_heading.group(1)),
                heading_normalized_text=stripped.strip(),
            )

        if _looks_like_uppercase_heading(stripped):
            return ClassifiedLine(
                classification=LineClassification.HEADING,
                heading_level=1,
                heading_normalized_text=normalize_text(stripped),
            )
        return ClassifiedLine(classification=LineClassification.PARAGRAPH)

    return scan_document(path, text, classify_line)
