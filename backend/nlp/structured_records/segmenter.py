"""
Structured record segmenter - converts parsed note documents into record units.

.. code-block:: mermaid

    flowchart TD
        A[ParsedMarkdownDocument] --> B[Collect headings in order]
        B --> C[Classify heading family]
        C --> D[Emit dossier_entry records]
        C --> E[Emit outline_beat records]
        C --> F[Emit reference_section records]
        B --> G[Detect pre-heading or no-heading content]
        G --> H[Emit loose_record fallback]
        D & E & F & H --> I[StructuredRecord list]
"""

from __future__ import annotations

import re

from backend.nlp.harvesting.shared import stable_hash_id
from backend.nlp.types import (
    Heading,
    ParsedMarkdownDocument,
    SpanAnchor,
    StructuredRecord,
    StructuredRecordType,
)

_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s*[—–-]\s+")
_ROMAN_HEADING_RE = re.compile(r"^\s*[IVXLCDM]+\.\s+")
_DASH_HEADING_RE = re.compile(r"[—–-]")
_LABEL_VALUE_LINE_RE = re.compile(r"(?m)^\s*[^:\n]{1,48}:\s+\S")


def _clean_heading_text(text: str) -> str:
    """Return a normalized heading text for heuristic checks.

    Args:
        text: Raw parsed heading text.

    Returns:
        Stripped heading text.
    """
    return text.strip()


def _alphabetic_upper_ratio(text: str) -> float:
    """Return the uppercase ratio across alphabetic characters.

    Args:
        text: Heading text to inspect.

    Returns:
        Ratio in the range [0.0, 1.0].
    """
    alpha_chars = [character for character in text if character.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for character in alpha_chars if character.isupper()) / len(alpha_chars)


def _looks_like_dossier_entry_heading(text: str) -> bool:
    """Return True when a heading looks like one dossier subject entry.

    Args:
        text: Cleaned heading text.

    Returns:
        True when the heading matches dossier-entry shape.
    """
    if _ROMAN_HEADING_RE.match(text):
        return False
    if not _DASH_HEADING_RE.search(text):
        return False
    if _NUMBERED_HEADING_RE.match(text):
        return False
    if _alphabetic_upper_ratio(text) < 0.65:
        return False
    parts = _DASH_HEADING_RE.split(text, maxsplit=1)
    if len(parts) < 2:
        return False
    left_side, right_side = parts[0].strip(), parts[1].strip()
    if ":" in left_side:
        return False
    if any(char.isdigit() for char in left_side):
        return False
    left_words = [word for word in re.split(r"\s+", left_side) if any(char.isalpha() for char in word)]
    if len(left_words) < 2:
        return False
    return "/" in right_side or any(char.isdigit() for char in right_side)


def _word_starts_with_uppercase(text: str) -> bool:
    """Return True when the first alphabetic character is uppercase.

    Args:
        text: One token or token-like fragment from a heading side.

    Returns:
        True when the first alphabetic character is uppercase.
    """
    for character in text:
        if character.isalpha():
            return character.isupper()
    return False


def _looks_like_inline_named_subject_heading(text: str) -> bool:
    """Return True for title-case subject headers embedded inside prose sections.

    Some planning files carry a large reference heading followed by repeated
    inline character headers such as ``Kurosawa Dia — Explorer (O-5)``. Those
    are real record boundaries, but they are not uppercase enough to pass the
    stricter dossier-heading heuristic used for the first crew-summary block.

    Args:
        text: Cleaned line text from inside one larger section.

    Returns:
        True when the line looks like an embedded subject heading rather than a
        sentence or banner.
    """
    if _looks_like_dossier_entry_heading(text):
        return True
    if _ROMAN_HEADING_RE.match(text):
        return False
    if _NUMBERED_HEADING_RE.match(text):
        return False
    if not _DASH_HEADING_RE.search(text):
        return False
    if text.endswith("."):
        return False

    parts = _DASH_HEADING_RE.split(text, maxsplit=1)
    if len(parts) < 2:
        return False
    left_side, right_side = parts[0].strip(), parts[1].strip()
    if not right_side or right_side.startswith(("“", "\"", "'")):
        return False
    if ":" in left_side:
        return False
    if any(char.isdigit() for char in left_side):
        return False

    uppercase_ratio = _alphabetic_upper_ratio(text)
    if uppercase_ratio >= 0.55:
        return False

    left_words = [word for word in re.split(r"\s+", left_side) if any(char.isalpha() for char in word)]
    if not 2 <= len(left_words) <= 4:
        return False
    if not all(_word_starts_with_uppercase(word) for word in left_words):
        return False
    return True


def _looks_like_banner_heading(text: str) -> bool:
    """Return True when a heading looks like a document or section banner.

    Args:
        text: Cleaned heading text.

    Returns:
        True when the heading is banner-like.
    """
    if _looks_like_dossier_entry_heading(text):
        return False
    if _NUMBERED_HEADING_RE.match(text):
        return False
    return _alphabetic_upper_ratio(text) >= 0.8


def _heading_record_type(heading: Heading) -> StructuredRecordType:
    """Classify a parsed heading into one structured-record family.

    Args:
        heading: Parsed heading span.

    Returns:
        Dominant record family for the heading.
    """
    text = _clean_heading_text(heading.text)
    if _looks_like_dossier_entry_heading(text):
        return StructuredRecordType.DOSSIER_ENTRY
    if _NUMBERED_HEADING_RE.match(text):
        return StructuredRecordType.OUTLINE_BEAT
    return StructuredRecordType.REFERENCE_SECTION


def _structural_flags(
    record_type: StructuredRecordType,
    heading_text: str,
    raw_text: str,
) -> list[str]:
    """Build deterministic structure flags for one segmented record.

    Args:
        record_type: Classified record family.
        heading_text: Heading text opening the record.
        raw_text: Full raw record text.

    Returns:
        Stable flag list for downstream review.
    """
    flags: list[str] = [f"record_type:{record_type.value}"]
    if heading_text and _looks_like_banner_heading(heading_text):
        flags.append("heading_like_uppercase")
    if _NUMBERED_HEADING_RE.match(heading_text):
        flags.append("numbered_outline_heading")
    if _looks_like_inline_named_subject_heading(heading_text):
        flags.append("dossier_subject_heading")
    if _LABEL_VALUE_LINE_RE.search(raw_text):
        flags.append("has_label_value_lines")
    if "\n-" in raw_text or "\n*" in raw_text:
        flags.append("has_bullets")
    if any(len(line.strip()) > 90 for line in raw_text.splitlines()):
        flags.append("has_long_prose_lines")
    return flags


def _record_anchor(path: str, span_ordinal: int, start_char: int, end_char: int) -> SpanAnchor:
    """Build a stable record anchor.

    Args:
        path: Source document path.
        span_ordinal: First span ordinal covered by the record.
        start_char: Inclusive start offset.
        end_char: Exclusive end offset.

    Returns:
        SpanAnchor for the segmented record.
    """
    return SpanAnchor(
        path=path,
        span_ordinal=span_ordinal,
        start_char=start_char,
        end_char=end_char,
    )


def _inline_record_type(text: str) -> StructuredRecordType | None:
    """Return an inline record family for one heading-shaped body line.

    Args:
        text: Cleaned line text from inside one larger section.

    Returns:
        Matching structured record type, or None when the line is not a record
        boundary.
    """
    if _looks_like_inline_named_subject_heading(text):
        return StructuredRecordType.DOSSIER_ENTRY
    if _ROMAN_HEADING_RE.match(text):
        return StructuredRecordType.REFERENCE_SECTION
    if _NUMBERED_HEADING_RE.match(text):
        return StructuredRecordType.OUTLINE_BEAT
    return None


def _inline_record_boundaries(
    raw_text: str,
    start_char: int,
    end_char: int,
) -> list[tuple[int, int, str, StructuredRecordType]]:
    """Return inline record boundaries inside one section-sized range.

    Args:
        raw_text: Full source document text.
        start_char: Inclusive section start offset.
        end_char: Exclusive section end offset.

    Returns:
        Tuples of ``(record_start, record_end, heading_text, record_type)``.
    """
    section_text = raw_text[start_char:end_char]
    position = start_char
    heading_positions: list[tuple[int, str, StructuredRecordType]] = []
    for line in section_text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r").strip()
        record_type = _inline_record_type(stripped) if stripped else None
        if record_type is not None:
            heading_positions.append((position, stripped, record_type))
        position += len(line)

    if not heading_positions:
        return []

    boundaries: list[tuple[int, int, str, StructuredRecordType]] = []
    for index, (entry_start, heading_text, record_type) in enumerate(heading_positions):
        entry_end = heading_positions[index + 1][0] if index + 1 < len(heading_positions) else end_char
        boundaries.append((entry_start, entry_end, heading_text, record_type))
    return boundaries


def segment_structured_records(doc: ParsedMarkdownDocument) -> list[StructuredRecord]:
    """Segment a parsed note document into structured record units.

    Args:
        doc: Parsed shared-span document.

    Returns:
        Structured records in document order.
    """
    headings = sorted(doc.headings, key=lambda heading: heading.start_char)
    if not headings:
        inline_record_boundaries = _inline_record_boundaries(doc.raw_text, 0, len(doc.raw_text))
        if inline_record_boundaries:
            records: list[StructuredRecord] = []
            all_spans = sorted([*doc.headings, *doc.paragraphs, *doc.scene_breaks], key=lambda span: span.span_ordinal)
            record_ordinal = 0
            first_record_start = inline_record_boundaries[0][0]
            if first_record_start > 0 and doc.raw_text[:first_record_start].strip():
                pre_text = doc.raw_text[:first_record_start]
                pre_span_ordinals = [
                    span.span_ordinal
                    for span in all_spans
                    if span.end_char <= first_record_start
                ]
                records.append(
                    StructuredRecord(
                        record_id=stable_hash_id(doc.path, StructuredRecordType.LOOSE_RECORD.value, "prelude"),
                        document_path=doc.path,
                        record_type=StructuredRecordType.LOOSE_RECORD,
                        anchor=_record_anchor(
                            doc.path,
                            pre_span_ordinals[0] if pre_span_ordinals else 0,
                            0,
                            first_record_start,
                        ),
                        start_char=0,
                        end_char=first_record_start,
                        raw_text=pre_text,
                        source_span_ordinals=pre_span_ordinals,
                        structural_flags=["record_type:loose_record", "pre_heading_content"],
                        ordinal_within_document=0,
                        field_lines=pre_text.splitlines(),
                    )
                )
                record_ordinal += 1

            for record_start, record_end, heading_text, record_type in inline_record_boundaries:
                record_text = doc.raw_text[record_start:record_end].rstrip()
                span_ordinals = [
                    span.span_ordinal
                    for span in all_spans
                    if record_start <= span.start_char and span.end_char <= record_end
                ]
                records.append(
                    StructuredRecord(
                        record_id=stable_hash_id(
                            doc.path,
                            record_type.value,
                            str(record_start),
                            heading_text,
                        ),
                        document_path=doc.path,
                        record_type=record_type,
                        anchor=_record_anchor(
                            doc.path,
                            span_ordinals[0] if span_ordinals else 0,
                            record_start,
                            record_end,
                        ),
                        start_char=record_start,
                        end_char=record_end,
                        heading_text=heading_text,
                        raw_text=record_text,
                        source_span_ordinals=span_ordinals,
                        structural_flags=_structural_flags(record_type, heading_text, record_text),
                        ordinal_within_document=record_ordinal,
                        field_lines=record_text.splitlines()[1:],
                        suspected_subject_line=heading_text if record_type == StructuredRecordType.DOSSIER_ENTRY else "",
                    )
                )
                record_ordinal += 1
            return records

        anchor = _record_anchor(doc.path, 0, 0, len(doc.raw_text))
        return [
            StructuredRecord(
                record_id=stable_hash_id(doc.path, StructuredRecordType.LOOSE_RECORD.value, "0"),
                document_path=doc.path,
                record_type=StructuredRecordType.LOOSE_RECORD,
                anchor=anchor,
                start_char=0,
                end_char=len(doc.raw_text),
                raw_text=doc.raw_text,
                source_span_ordinals=[
                    span.span_ordinal
                    for span in sorted(
                        [*doc.headings, *doc.paragraphs, *doc.scene_breaks],
                        key=lambda span: span.span_ordinal,
                    )
                ],
                structural_flags=["record_type:loose_record", "no_headings_detected"],
                ordinal_within_document=0,
                field_lines=doc.raw_text.splitlines(),
            )
        ]

    records: list[StructuredRecord] = []
    all_spans = sorted([*doc.headings, *doc.paragraphs, *doc.scene_breaks], key=lambda span: span.span_ordinal)

    if headings[0].start_char > 0 and doc.raw_text[:headings[0].start_char].strip():
        pre_text = doc.raw_text[:headings[0].start_char]
        pre_span_ordinals = [
            span.span_ordinal
            for span in all_spans
            if span.end_char <= headings[0].start_char
        ]
        pre_anchor = _record_anchor(doc.path, pre_span_ordinals[0] if pre_span_ordinals else 0, 0, headings[0].start_char)
        records.append(
            StructuredRecord(
                record_id=stable_hash_id(doc.path, StructuredRecordType.LOOSE_RECORD.value, "prelude"),
                document_path=doc.path,
                record_type=StructuredRecordType.LOOSE_RECORD,
                anchor=pre_anchor,
                start_char=0,
                end_char=headings[0].start_char,
                raw_text=pre_text,
                source_span_ordinals=pre_span_ordinals,
                structural_flags=["record_type:loose_record", "pre_heading_content"],
                ordinal_within_document=0,
                field_lines=pre_text.splitlines(),
            )
        )

    last_banner_heading = ""
    record_ordinal = len(records)
    for index, heading in enumerate(headings):
        heading_text = _clean_heading_text(heading.text)
        if _looks_like_banner_heading(heading_text):
            last_banner_heading = heading_text

        record_type = _heading_record_type(heading)
        next_heading_start = headings[index + 1].start_char if index + 1 < len(headings) else len(doc.raw_text)
        raw_text = doc.raw_text[heading.start_char:next_heading_start].rstrip()
        if not raw_text.strip():
            continue

        inline_record_boundaries = _inline_record_boundaries(
            doc.raw_text,
            heading.start_char,
            next_heading_start,
        )
        if record_type != StructuredRecordType.DOSSIER_ENTRY and inline_record_boundaries:
            for entry_start, entry_end, inline_heading_text, inline_record_type in inline_record_boundaries:
                entry_text = doc.raw_text[entry_start:entry_end].rstrip()
                span_ordinals = [
                    span.span_ordinal
                    for span in all_spans
                    if entry_start <= span.start_char and span.end_char <= entry_end
                ]
                records.append(
                    StructuredRecord(
                        record_id=stable_hash_id(
                            doc.path,
                            inline_record_type.value,
                            str(entry_start),
                            inline_heading_text,
                        ),
                        document_path=doc.path,
                        record_type=inline_record_type,
                        anchor=_record_anchor(
                            doc.path,
                            span_ordinals[0] if span_ordinals else heading.span_ordinal,
                            entry_start,
                            entry_end,
                        ),
                        start_char=entry_start,
                        end_char=entry_end,
                        heading_text=inline_heading_text,
                        raw_text=entry_text,
                        source_span_ordinals=span_ordinals,
                        structural_flags=_structural_flags(
                            inline_record_type,
                            inline_heading_text,
                            entry_text,
                        ),
                        parent_heading=heading_text,
                        ordinal_within_document=record_ordinal,
                        field_lines=entry_text.splitlines()[1:],
                        suspected_subject_line=(
                            inline_heading_text
                            if inline_record_type == StructuredRecordType.DOSSIER_ENTRY
                            else ""
                        ),
                    )
                )
                record_ordinal += 1
            continue

        span_ordinals = [
            span.span_ordinal
            for span in all_spans
            if heading.start_char <= span.start_char and span.end_char <= next_heading_start
        ]
        records.append(
            StructuredRecord(
                record_id=stable_hash_id(doc.path, record_type.value, str(heading.start_char), heading_text),
                document_path=doc.path,
                record_type=record_type,
                anchor=_record_anchor(
                    doc.path,
                    span_ordinals[0] if span_ordinals else heading.span_ordinal,
                    heading.start_char,
                    next_heading_start,
                ),
                start_char=heading.start_char,
                end_char=next_heading_start,
                heading_text=heading_text,
                raw_text=raw_text,
                source_span_ordinals=span_ordinals,
                structural_flags=_structural_flags(record_type, heading_text, raw_text),
                parent_heading=last_banner_heading if last_banner_heading != heading_text else "",
                ordinal_within_document=record_ordinal,
                field_lines=raw_text.splitlines()[1:],
                suspected_subject_line=heading_text if record_type == StructuredRecordType.DOSSIER_ENTRY else "",
            )
        )
        record_ordinal += 1

    return records
