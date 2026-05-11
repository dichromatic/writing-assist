"""
Structured record seed extractor - builds deterministic seed packets by record type.

.. code-block:: mermaid

    flowchart TD
        A[StructuredRecord] --> B[Scan header and body lines]
        B --> C[Classify field-like lines]
        A --> D[Filter overlapping entity records]
        A --> E[Filter overlapping reference candidates]
        C & D & E --> F[Dispatch by record type]
        F --> G[Build DeterministicSeedBundle]
        G --> H[Build subject guess and fact candidates]
"""

from __future__ import annotations

import re

from backend.nlp.types import (
    DeterministicFactCandidate,
    DeterministicGuess,
    DeterministicSeedBundle,
    DocumentEntityRecord,
    ReferenceCandidate,
    StructuredFieldLine,
    StructuredFieldLineType,
    StructuredRecord,
    StructuredRecordType,
)

_LABEL_VALUE_RE = re.compile(r"^\s*([^:]{1,48}):\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_SUBHEAD_RE = re.compile(r"^[A-Z][A-Za-z0-9&/'() \-]{1,60}$")
_HEADER_SPLIT_RE = re.compile(r"\s*[—–-]\s*")
_SCENE_BREAK_TEXTS = {"---", "***", "___"}


def _overlaps_record(record: StructuredRecord, start_char: int, end_char: int) -> bool:
    """Return True when a source span overlaps a structured record.

    Args:
        record: Structured record boundary.
        start_char: Candidate inclusive start offset.
        end_char: Candidate exclusive end offset.

    Returns:
        True when the candidate lies inside or overlaps the record.
    """
    return record.start_char < end_char and start_char < record.end_char


def _field_lines(record: StructuredRecord) -> list[StructuredFieldLine]:
    """Classify raw body lines into shallow structural field groups.

    Args:
        record: Structured record whose body lines will be scanned.

    Returns:
        Structured field lines in original order.
    """
    grouped_lines: list[StructuredFieldLine] = []
    for line_index, raw_line in enumerate(record.field_lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped in _SCENE_BREAK_TEXTS:
            continue
        label_match = _LABEL_VALUE_RE.match(stripped)
        if label_match:
            grouped_lines.append(StructuredFieldLine(
                line_index=line_index,
                line_type=StructuredFieldLineType.LABEL_VALUE,
                raw_text=stripped,
                label=label_match.group(1).strip(),
                value=label_match.group(2).strip(),
            ))
            continue
        if _BULLET_RE.match(stripped):
            grouped_lines.append(StructuredFieldLine(
                line_index=line_index,
                line_type=StructuredFieldLineType.BULLET,
                raw_text=stripped,
            ))
            continue
        if _SUBHEAD_RE.match(stripped) and not stripped.endswith("."):
            grouped_lines.append(StructuredFieldLine(
                line_index=line_index,
                line_type=StructuredFieldLineType.STANDALONE_SUBHEAD,
                raw_text=stripped,
            ))
            continue
        grouped_lines.append(StructuredFieldLine(
            line_index=line_index,
            line_type=StructuredFieldLineType.PROSE,
            raw_text=stripped,
        ))
    return grouped_lines


def _dossier_subject_guess(record: StructuredRecord) -> DeterministicGuess | None:
    """Build a conservative dossier subject guess from the header line.

    Args:
        record: Structured dossier record.

    Returns:
        A non-final deterministic subject guess, or None when no guess is
        justified.
    """
    if not record.suspected_subject_line:
        return None
    header = record.suspected_subject_line.strip()
    left_side = _HEADER_SPLIT_RE.split(header, maxsplit=1)[0]
    cleaned = re.sub(r"^[^\wA-Za-z]+", "", left_side).strip()
    if not cleaned:
        return None
    return DeterministicGuess(
        guess_type="subject",
        primary_guess=cleaned.title(),
        alternative_guesses=[],
        reason="header-shaped dossier subject line",
        supporting_anchor=record.anchor,
    )


def _dossier_candidate_rank_texts(record: StructuredRecord) -> list[str]:
    """Extract raw rank-like or title-like fragments from one dossier header.

    Args:
        record: Structured dossier record.

    Returns:
        Rank or title strings preserved from the right side of the header.
    """
    if not record.heading_text or len(_HEADER_SPLIT_RE.split(record.heading_text, maxsplit=1)) < 2:
        return []
    right_side = _HEADER_SPLIT_RE.split(record.heading_text, maxsplit=1)[1]
    return [
        part.strip()
        for part in right_side.split("/")
        if part.strip()
    ]


def _dossier_fact_candidates(
    record: StructuredRecord,
    grouped_lines: list[StructuredFieldLine],
    candidate_rank_texts: list[str],
) -> list[DeterministicFactCandidate]:
    """Build shallow deterministic fact-like candidates from one record.

    Args:
        record: Structured dossier record.
        grouped_lines: Structured field lines already classified.
        candidate_rank_texts: Header-derived title or rank strings.

    Returns:
        Fact-like candidates in stable order.
    """
    fact_candidates: list[DeterministicFactCandidate] = []
    for rank_text in candidate_rank_texts:
        fact_candidates.append(DeterministicFactCandidate(
            label="header_rank",
            value=rank_text,
            reason="header right side preserved as rank or title evidence",
            supporting_anchor=record.anchor,
        ))
    for field_line in grouped_lines:
        if field_line.line_type != StructuredFieldLineType.LABEL_VALUE:
            continue
        fact_candidates.append(DeterministicFactCandidate(
            label=field_line.label,
            value=field_line.value,
            reason="label-value line preserved as explicit dossier field",
            supporting_anchor=record.anchor,
            line_index=field_line.line_index,
        ))
    return fact_candidates


def _generic_heading_fact(
    record: StructuredRecord,
    *,
    label: str,
    reason: str,
) -> list[DeterministicFactCandidate]:
    """Return one heading-backed fact candidate when the record has a heading.

    Args:
        record: Structured record being scanned.
        label: Fact label to use for the heading text.
        reason: Why the heading is being preserved as a fact-like candidate.

    Returns:
        A one-item list when the heading exists, otherwise an empty list.
    """
    if not record.heading_text:
        return []
    return [DeterministicFactCandidate(
        label=label,
        value=record.heading_text,
        reason=reason,
        supporting_anchor=record.anchor,
    )]


def _reference_section_fact_candidates(
    record: StructuredRecord,
    grouped_lines: list[StructuredFieldLine],
) -> list[DeterministicFactCandidate]:
    """Build conservative fact-like candidates for reference sections.

    Reference sections are explanatory by nature. Preserve the section heading
    and each explicit line shape so later passes can normalize them without
    reparsing the raw note.

    Args:
        record: Structured reference-section record.
        grouped_lines: Structured field lines already classified.

    Returns:
        Fact-like candidates in stable order.
    """
    fact_candidates = _generic_heading_fact(
        record,
        label="section_heading",
        reason="reference-section heading preserved as explicit section context",
    )
    for field_line in grouped_lines:
        if field_line.line_type == StructuredFieldLineType.LABEL_VALUE:
            fact_candidates.append(DeterministicFactCandidate(
                label=field_line.label,
                value=field_line.value,
                reason="label-value line preserved as explicit section field",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.STANDALONE_SUBHEAD:
            fact_candidates.append(DeterministicFactCandidate(
                label="section_subheading",
                value=field_line.raw_text,
                reason="standalone subheading preserved as internal section structure",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.BULLET:
            fact_candidates.append(DeterministicFactCandidate(
                label="section_bullet",
                value=_BULLET_RE.sub("", field_line.raw_text).strip(),
                reason="bullet line preserved as explicit list evidence inside a reference section",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        else:
            fact_candidates.append(DeterministicFactCandidate(
                label="section_prose",
                value=field_line.raw_text,
                reason="prose line preserved as explicit section statement",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
    return fact_candidates


def _outline_beat_fact_candidates(
    record: StructuredRecord,
    grouped_lines: list[StructuredFieldLine],
) -> list[DeterministicFactCandidate]:
    """Build conservative fact-like candidates for outline beats.

    Outline beats are action-oriented and often mix bullets with short prose
    notes. Preserve the beat heading and each line shape so later passes can
    infer event and participant structure from explicit planning language.

    Args:
        record: Structured outline-beat record.
        grouped_lines: Structured field lines already classified.

    Returns:
        Fact-like candidates in stable order.
    """
    fact_candidates = _generic_heading_fact(
        record,
        label="beat_heading",
        reason="outline-beat heading preserved as explicit planning milestone",
    )
    for field_line in grouped_lines:
        if field_line.line_type == StructuredFieldLineType.LABEL_VALUE:
            fact_candidates.append(DeterministicFactCandidate(
                label=field_line.label,
                value=field_line.value,
                reason="label-value line preserved as explicit planning field",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.STANDALONE_SUBHEAD:
            fact_candidates.append(DeterministicFactCandidate(
                label="beat_subheading",
                value=field_line.raw_text,
                reason="standalone subheading preserved as internal beat structure",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.BULLET:
            fact_candidates.append(DeterministicFactCandidate(
                label="beat_step",
                value=_BULLET_RE.sub("", field_line.raw_text).strip(),
                reason="bullet line preserved as explicit beat step",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        else:
            fact_candidates.append(DeterministicFactCandidate(
                label="beat_note",
                value=field_line.raw_text,
                reason="prose line preserved as explicit beat note",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
    return fact_candidates


def _loose_record_fact_candidates(
    record: StructuredRecord,
    grouped_lines: list[StructuredFieldLine],
) -> list[DeterministicFactCandidate]:
    """Build conservative fact-like candidates for loose records.

    Loose records are not failed extraction. They preserve note fragments that
    do not have enough structure for a stronger family yet, so later review and
    LLM passes can still use the material without reparsing raw prose.

    Args:
        record: Structured loose record.
        grouped_lines: Structured field lines already classified.

    Returns:
        Fact-like candidates in stable order.
    """
    fact_candidates: list[DeterministicFactCandidate] = []
    if record.label_text:
        fact_candidates.append(DeterministicFactCandidate(
            label="loose_label",
            value=record.label_text,
            reason="loose-record label preserved as weak structural context",
            supporting_anchor=record.anchor,
        ))
    for field_line in grouped_lines:
        if field_line.line_type == StructuredFieldLineType.LABEL_VALUE:
            fact_candidates.append(DeterministicFactCandidate(
                label=field_line.label,
                value=field_line.value,
                reason="label-value line preserved inside loose record",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.STANDALONE_SUBHEAD:
            fact_candidates.append(DeterministicFactCandidate(
                label="loose_subheading",
                value=field_line.raw_text,
                reason="standalone line preserved as weak loose-record structure",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        elif field_line.line_type == StructuredFieldLineType.BULLET:
            fact_candidates.append(DeterministicFactCandidate(
                label="loose_bullet",
                value=_BULLET_RE.sub("", field_line.raw_text).strip(),
                reason="bullet line preserved inside loose record",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
        else:
            fact_candidates.append(DeterministicFactCandidate(
                label="loose_note",
                value=field_line.raw_text,
                reason="prose line preserved inside loose record",
                supporting_anchor=record.anchor,
                line_index=field_line.line_index,
            ))
    return fact_candidates


def build_record_seed_bundle(
    record: StructuredRecord,
    entity_records: list[DocumentEntityRecord],
    reference_candidates: list[ReferenceCandidate],
) -> tuple[DeterministicSeedBundle, DeterministicGuess | None, list[DeterministicFactCandidate]]:
    """Build the deterministic seed packet for one supported structured record.

    Args:
        record: Structured record to seed.
        entity_records: Whole-document entity summaries used as weak hints.
        reference_candidates: Whole-document deferred references used as weak
            hints.

    Returns:
        The deterministic seed bundle, subject guess, and fact candidates.
    """
    grouped_lines = _field_lines(record)
    if record.record_type == StructuredRecordType.DOSSIER_ENTRY:
        subject_guess = _dossier_subject_guess(record)
        rank_texts = _dossier_candidate_rank_texts(record)
        fact_candidates = _dossier_fact_candidates(record, grouped_lines, rank_texts)
    elif record.record_type == StructuredRecordType.REFERENCE_SECTION:
        subject_guess = None
        rank_texts = []
        fact_candidates = _reference_section_fact_candidates(record, grouped_lines)
    elif record.record_type == StructuredRecordType.OUTLINE_BEAT:
        subject_guess = None
        rank_texts = []
        fact_candidates = _outline_beat_fact_candidates(record, grouped_lines)
    elif record.record_type == StructuredRecordType.LOOSE_RECORD:
        subject_guess = None
        rank_texts = []
        fact_candidates = _loose_record_fact_candidates(record, grouped_lines)
    else:
        subject_guess = None
        rank_texts = []
        fact_candidates = []

    overlapping_entities = [
        entity_record
        for entity_record in entity_records
        if any(
            _overlaps_record(record, anchor.start_char, anchor.end_char)
            for anchor in entity_record.anchors
        )
    ]
    overlapping_references = [
        reference_candidate
        for reference_candidate in reference_candidates
        if _overlaps_record(
            record,
            reference_candidate.anchor.start_char,
            reference_candidate.anchor.end_char,
        )
    ]
    known_canon_matches = [
        entity_record.normalized_key
        for entity_record in sorted(
            overlapping_entities,
            key=lambda item: (-item.confidence_score, item.normalized_key),
        )
    ]
    bundle = DeterministicSeedBundle(
        record_id=record.record_id,
        header_line=record.heading_text,
        suspected_subject_guess=subject_guess,
        candidate_rank_texts=rank_texts,
        field_lines=grouped_lines,
        entity_candidates=sorted(
            overlapping_entities,
            key=lambda item: (item.bucket.value, -item.confidence_score, item.normalized_key),
        ),
        reference_candidates=sorted(
            overlapping_references,
            key=lambda item: (
                item.anchor.start_char,
                item.reference_type.value,
                item.normalized,
            ),
        ),
        known_canon_matches=known_canon_matches,
        structural_flags=list(record.structural_flags),
    )
    return bundle, subject_guess, fact_candidates


def build_dossier_seed_bundle(
    record: StructuredRecord,
    entity_records: list[DocumentEntityRecord],
    reference_candidates: list[ReferenceCandidate],
) -> tuple[DeterministicSeedBundle, DeterministicGuess | None, list[DeterministicFactCandidate]]:
    """Build the deterministic dossier seed packet for one structured record.

    Args:
        record: Structured record to seed.
        entity_records: Whole-document entity summaries used as weak hints.
        reference_candidates: Whole-document deferred references used as weak
            hints.

    Returns:
        The deterministic seed bundle, subject guess, and fact candidates.
    """
    return build_record_seed_bundle(record, entity_records, reference_candidates)
