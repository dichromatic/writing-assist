"""
Document metadata resolver - attaches source authority metadata to corpus files.

Document metadata is provenance, not routing. It can influence later review and
retrieval weighting, but it must not choose the record family or segmentation
pipeline.

.. code-block:: mermaid

    flowchart TD
        A[Document path plus raw text] --> B[Classify document type]
        A --> C[Read in-document status]
        D[Optional sidecar manifest] --> E[Read sidecar status]
        A --> F[Collect folder hints]
        C & E --> G{Explicit status conflict?}
        G -->|Yes| H[draft_unknown plus conflict]
        G -->|No| I[Explicit status or default primary_canon]
        B & F & H & I --> J[DocumentMetadata]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.nlp.document_type import classify_document_type
from backend.nlp.types import DocumentMetadata, DocumentStatus

_STATUS_LINE_RE = re.compile(
    r"^\s*(?:document[_ -]?status|canon[_ -]?status|status)\s*:\s*(?P<value>[\"']?[A-Za-z_ -]+[\"']?)\s*$",
    re.IGNORECASE,
)
_STATUS_ALIASES = {
    "primary_canon": DocumentStatus.PRIMARY_CANON,
    "primary canon": DocumentStatus.PRIMARY_CANON,
    "canon": DocumentStatus.PRIMARY_CANON,
    "canonical": DocumentStatus.PRIMARY_CANON,
    "historical": DocumentStatus.HISTORICAL,
    "history": DocumentStatus.HISTORICAL,
    "legendary": DocumentStatus.LEGENDARY,
    "legend": DocumentStatus.LEGENDARY,
    "draft_unknown": DocumentStatus.DRAFT_UNKNOWN,
    "draft unknown": DocumentStatus.DRAFT_UNKNOWN,
    "draft": DocumentStatus.DRAFT_UNKNOWN,
    "unknown": DocumentStatus.DRAFT_UNKNOWN,
    "apocryphal": DocumentStatus.APOCRYPHAL,
    "apocrypha": DocumentStatus.APOCRYPHAL,
}


def _normalise_status(value: str) -> DocumentStatus | None:
    """Normalize a free-text status value into the controlled status set.

    Args:
        value: Raw status string.

    Returns:
        Matching document status, or None when the value is unsupported.
    """
    normalized = value.strip().casefold().replace("-", "_")
    return _STATUS_ALIASES.get(normalized) or _STATUS_ALIASES.get(normalized.replace("_", " "))


def _status_from_text(raw_text: str) -> tuple[DocumentStatus | None, str]:
    """Read an explicit status line from the start of a document.

    Args:
        raw_text: Raw source document text.

    Returns:
        Resolved status plus the raw matched value, if supported.
    """
    for line in raw_text.splitlines()[:40]:
        match = _STATUS_LINE_RE.match(line)
        if match is None:
            continue
        raw_value = match.group("value").strip().strip("\"'")
        status = _normalise_status(raw_value)
        if status is not None:
            return status, raw_value
    return None, ""


def _manifest_documents(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the document mapping from a supported manifest shape.

    Args:
        manifest: Decoded sidecar manifest.

    Returns:
        Mapping from document path or filename to metadata values.
    """
    documents = manifest.get("documents")
    if isinstance(documents, dict):
        return documents
    return manifest


def _status_from_manifest(path: str, manifest: dict[str, Any] | None) -> tuple[DocumentStatus | None, str]:
    """Read a status for a path from an optional sidecar manifest.

    Args:
        path: Source document path.
        manifest: Optional decoded sidecar manifest.

    Returns:
        Resolved status plus the manifest key that supplied it.
    """
    if manifest is None:
        return None, ""
    source_path = Path(path)
    documents = _manifest_documents(manifest)
    lookup_keys = [
        path,
        source_path.as_posix(),
        source_path.name,
        source_path.stem,
    ]
    for key in lookup_keys:
        raw_entry = documents.get(key)
        if raw_entry is None:
            continue
        raw_status = raw_entry.get("status") if isinstance(raw_entry, dict) else raw_entry
        if not isinstance(raw_status, str):
            continue
        status = _normalise_status(raw_status)
        if status is not None:
            return status, key
    return None, ""


def _folder_hints(path: str) -> list[str]:
    """Return non-authoritative folder hints for review.

    Args:
        path: Source document path.

    Returns:
        Folder-derived hints that must not override explicit or default status.
    """
    hint_parts = {"vignettes", "history", "legends", "legendary", "drafts", "apocrypha"}
    hints: list[str] = []
    for part in Path(path).parts:
        normalized = part.casefold()
        if normalized in hint_parts:
            hints.append(f"folder:{part}")
    return hints


def load_document_metadata_manifest(path: str | None) -> dict[str, Any] | None:
    """Load an optional JSON sidecar manifest.

    Args:
        path: Manifest path, or None when no manifest was supplied.

    Returns:
        Decoded manifest mapping, or None.
    """
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_document_metadata(
    path: str,
    raw_text: str = "",
    sidecar_manifest: dict[str, Any] | None = None,
) -> DocumentMetadata:
    """Resolve document type and status metadata for one corpus file.

    Args:
        path: Source document path.
        raw_text: Raw source document text used for in-document metadata.
        sidecar_manifest: Optional decoded sidecar metadata manifest.

    Returns:
        Resolved document metadata. Explicit sidecar and in-document conflicts
        downgrade status to draft_unknown and surface a reviewable conflict.
    """
    document_type = classify_document_type(path)
    text_status, text_value = _status_from_text(raw_text)
    manifest_status, manifest_key = _status_from_manifest(path, sidecar_manifest)
    hints = _folder_hints(path)

    if text_status is not None and manifest_status is not None and text_status != manifest_status:
        return DocumentMetadata(
            document_path=path,
            document_type=document_type,
            document_status=DocumentStatus.DRAFT_UNKNOWN,
            status_source="conflict",
            status_hints=hints,
            metadata_conflicts=[
                "document_status mismatch: "
                f"in_document={text_status.value} sidecar={manifest_status.value}"
            ],
        )

    if manifest_status is not None:
        return DocumentMetadata(
            document_path=path,
            document_type=document_type,
            document_status=manifest_status,
            status_source=f"sidecar:{manifest_key}",
            status_hints=hints,
        )

    if text_status is not None:
        return DocumentMetadata(
            document_path=path,
            document_type=document_type,
            document_status=text_status,
            status_source=f"in_document:{text_value}",
            status_hints=hints,
        )

    return DocumentMetadata(
        document_path=path,
        document_type=document_type,
        document_status=DocumentStatus.PRIMARY_CANON,
        status_source="default",
        status_hints=hints,
    )


def document_status_authority_weight(status: DocumentStatus) -> float:
    """Return retrieval authority weight implied by document status.

    Args:
        status: Resolved document status.

    Returns:
        Authority weight in the range [0.0, 1.0].
    """
    if status == DocumentStatus.PRIMARY_CANON:
        return 1.0
    if status == DocumentStatus.HISTORICAL:
        return 0.85
    if status == DocumentStatus.LEGENDARY:
        return 0.75
    if status == DocumentStatus.DRAFT_UNKNOWN:
        return 0.5
    if status == DocumentStatus.APOCRYPHAL:
        return 0.25
    return 0.5
