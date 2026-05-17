"""
Structured entity extraction - builds deterministic entity mentions from record structure.

.. code-block:: mermaid

    flowchart TD
        A[StructuredRecord list] --> B[Pass 1: dossier subject and rank mentions]
        B --> C[StructuredEntityInventory]
"""

from __future__ import annotations

from collections import defaultdict
import re

from backend.nlp.structured_records.seed_extractor import _HEADER_SPLIT_RE
from backend.nlp.types import (
    StructuredEntityInventory,
    StructuredEntityMention,
    StructuredEntitySource,
    StructuredRecord,
    StructuredRecordType,
)

def _normalize_name(value: str) -> str:
    """Normalize a name candidate for inventory matching."""
    cleaned = re.sub(r"\s+", " ", value.strip())
    return cleaned.casefold()


def _clean_subject_or_rank(value: str) -> str:
    """Remove surrounding punctuation and compact whitespace."""
    cleaned = value.strip()
    cleaned = re.sub(r"^[^\wA-Za-z]+", "", cleaned)
    cleaned = re.sub(r"[^\w)\]]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_titleish_fragment(value: str) -> bool:
    """Return True when a fragment looks like a name-ish relationship label."""
    tokens = [token for token in re.split(r"\s+", value.strip()) if token]
    if not tokens:
        return False
    has_cased_token = False
    for token in tokens:
        alpha = re.sub(r"[^A-Za-z]", "", token)
        if not alpha:
            continue
        has_cased_token = True
        if not alpha[0].isupper():
            return False
    return has_cased_token


def _extract_dossier_subjects_and_ranks(records: list[StructuredRecord]) -> list[StructuredEntityMention]:
    """Extract subject-header and rank mentions from dossier records."""
    mentions: list[StructuredEntityMention] = []
    for record in records:
        if record.record_type != StructuredRecordType.DOSSIER_ENTRY:
            continue
        header = (record.suspected_subject_line or record.heading_text).strip()
        if not header:
            continue
        parts = _HEADER_SPLIT_RE.split(header, maxsplit=1)
        left_side = _clean_subject_or_rank(parts[0])
        if left_side:
            mentions.append(StructuredEntityMention(
                # Preserve author-provided casing from structured headers.
                # Title-casing is lossy for mixed-case names and acronyms.
                name=left_side,
                normalized_name=_normalize_name(left_side),
                source=StructuredEntitySource.SUBJECT_HEADER,
                anchor=record.anchor,
                record_id=record.record_id,
                document_path=record.document_path,
                source_label=header,
            ))
        if len(parts) < 2:
            continue
        right_side = parts[1]
        for part in right_side.split("/"):
            rank = _clean_subject_or_rank(part)
            if not rank:
                continue
            mentions.append(StructuredEntityMention(
                # Preserve original structured casing for rank and title text.
                name=rank,
                normalized_name=_normalize_name(rank),
                source=StructuredEntitySource.RANK_TEXT,
                anchor=record.anchor,
                record_id=record.record_id,
                document_path=record.document_path,
                source_label=header,
            ))
    return mentions


def _sort_mentions(records: list[StructuredRecord], mentions: list[StructuredEntityMention]) -> list[StructuredEntityMention]:
    """Sort mentions in deterministic document order."""
    ordinal_by_record = {record.record_id: record.ordinal_within_document for record in records}
    return sorted(
        mentions,
        key=lambda item: (
            item.document_path,
            ordinal_by_record.get(item.record_id, 10_000_000),
            item.anchor.start_char,
            item.name.casefold(),
            item.source.value,
        ),
    )


def extract_structural_entities(records: list[StructuredRecord]) -> StructuredEntityInventory:
    """Extract deterministic structured entities from one segmented document.

    The deterministic boundary for reference extraction is high-precision
    dossier header structure. Relationship-key and inventory matching stages
    are intentionally excluded to avoid semantic misclassification from
    author-specific metadata labels.
    """
    all_mentions = _sort_mentions(records, _extract_dossier_subjects_and_ranks(records))

    mentions_by_record: dict[str, list[StructuredEntityMention]] = defaultdict(list)
    records_by_name_set: dict[str, set[str]] = defaultdict(set)
    for mention in all_mentions:
        mentions_by_record[mention.record_id].append(mention)
        records_by_name_set[mention.normalized_name].add(mention.record_id)

    records_by_name = {
        name: sorted(record_ids)
        for name, record_ids in records_by_name_set.items()
    }
    return StructuredEntityInventory(
        mentions=all_mentions,
        names=frozenset(records_by_name.keys()),
        mentions_by_record=dict(mentions_by_record),
        records_by_name=records_by_name,
    )
