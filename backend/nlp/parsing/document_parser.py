"""
Document parser dispatcher - routes source files to the appropriate parser.

This boundary keeps the rest of the NLP pipeline format-agnostic. Downstream
stages receive the same parsed span model regardless of whether the source was
Markdown prose or plain-text notes.

.. code-block:: mermaid

    flowchart TD
        A[Path plus raw text] --> B{File suffix}
        B -->|.md / .markdown| C[Markdown parser]
        B -->|.txt| D[Text parser]
        B -->|Other| C
        C --> E[Parsed span model]
        D --> E
"""

from __future__ import annotations

from pathlib import Path

from backend.nlp.parsing.markdown_parser import parse as parse_markdown
from backend.nlp.parsing.text_parser import parse as parse_text
from backend.nlp.types import ParsedMarkdownDocument


def parse(path: str, text: str) -> ParsedMarkdownDocument:
    """Parse a source document using the parser implied by its suffix.

    Args:
        path: Source path used for suffix detection and anchor construction.
        text: Raw source document text.

    Returns:
        A parsed document in the shared span model.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return parse_text(path, text)
    return parse_markdown(path, text)
