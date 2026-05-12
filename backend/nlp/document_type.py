"""
Document type classifier - classifies corpus files before record segmentation.

.. code-block:: mermaid

    flowchart TD
        A[Corpus file path] --> B[Normalize path parts]
        B --> C{Known examples folder}
        C -->|story planning| D[story_planning]
        C -->|world context| E[world_context]
        C -->|vignettes| F[vignette]
        C -->|locations| G[location]
        C -->|character backgrounds| H[character_background]
        C -->|top-level .md example| I[manuscript]
        C -->|unknown| J[unknown]
"""

from __future__ import annotations

from pathlib import Path

from backend.nlp.types import DocumentType


def classify_document_type(path: str) -> DocumentType:
    """Classify a corpus file path into a document type.

    Args:
        path: Source corpus file path.

    Returns:
        Best current document type classification.
    """
    source_path = Path(path)
    normalized_parts = [part.casefold() for part in source_path.parts]
    if "story planning" in normalized_parts:
        return DocumentType.STORY_PLANNING
    if "world context" in normalized_parts:
        return DocumentType.WORLD_CONTEXT
    if "vignettes" in normalized_parts:
        return DocumentType.VIGNETTE
    if "locations" in normalized_parts:
        return DocumentType.LOCATION
    if "character backgrounds" in normalized_parts:
        return DocumentType.CHARACTER_BACKGROUND
    if (
        source_path.parent.name.casefold() == "examples"
        and source_path.suffix.casefold() in {".md", ".markdown"}
    ):
        return DocumentType.MANUSCRIPT
    return DocumentType.UNKNOWN
