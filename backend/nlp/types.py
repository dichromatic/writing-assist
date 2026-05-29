"""
Pipeline type definitions for the NLP evidence pipeline.

All input/output types for every stage live here. No stage module imports
from another stage module - they all import from this file. That keeps the
dependency graph a star: types.py at the centre, stage modules at the edges.

Diagram omitted - pure data model with no information flow to represent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Bumped when any field name or type changes in a breaking way. Intended for
# future serialization compatibility checks; not used at runtime yet.
PIPELINE_VERSION = "1"


# ---------------------------------------------------------------------------
# Stable identity
# ---------------------------------------------------------------------------

def stable_hash_id(*components: str) -> str:
    """Return a 16-character hex ID that is stable for the given components.

    Concatenates all components with a null-byte separator before hashing so
    that ("ab", "c") and ("a", "bc") cannot collide.

    Args:
        *components: One or more strings that together uniquely identify the
            record. Typically: document path, span ordinal, surface form.

    Returns:
        16-character lowercase hex string derived from SHA-256.
    """
    joined = "\x00".join(components)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Anchor types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentAnchor:
    """Identifies a document. Carried on every bundle-level record."""

    path: str


@dataclass(frozen=True)
class SpanAnchor:
    """Identifies a specific span within a document.

    Char offsets are relative to the raw document text. span_ordinal is the
    0-based position of the span in document order across all span types.

    Args:
        path: Document path.
        span_ordinal: Position of the span in the document, 0-based.
        start_char: Inclusive start character offset in the raw document text.
        end_char: Exclusive end character offset in the raw document text.
    """

    path: str
    span_ordinal: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SectionAnchor:
    """Identifies a section within a document.

    Args:
        path: Document path.
        section_index: 0-based index of the section in document order.
    """

    path: str
    section_index: int


# ---------------------------------------------------------------------------
# Parsing stage output (markdown_parser.py)
# ---------------------------------------------------------------------------

@dataclass
class Heading:
    """A Markdown ATX heading span.

    Args:
        text: Raw heading text including the leading # characters.
        level: Heading depth (1-6).
        normalized_text: Whitespace-collapsed text with # stripped, for retrieval.
        span_ordinal: Position in document order.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        anchor: Source anchor for this span.
    """

    text: str
    level: int
    normalized_text: str
    span_ordinal: int
    start_char: int
    end_char: int
    anchor: SpanAnchor


@dataclass
class Paragraph:
    """A prose paragraph span.

    Args:
        text: Raw paragraph text.
        normalized_text: Whitespace-collapsed text for retrieval.
        span_ordinal: Position in document order.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        anchor: Source anchor for this span.
    """

    text: str
    normalized_text: str
    span_ordinal: int
    start_char: int
    end_char: int
    anchor: SpanAnchor


@dataclass
class SceneBreak:
    """An explicit scene-break marker on its own line.

    Recognised forms are ---, ***, and ___ (three or more of each character),
    matching the CommonMark thematic break specification.

    Args:
        span_ordinal: Position in document order.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        anchor: Source anchor for this span.
    """

    span_ordinal: int
    start_char: int
    end_char: int
    anchor: SpanAnchor


@dataclass
class Section:
    """A document section bounded by headings.

    Sections are derived from heading boundaries, not emitted as raw spans.
    The first section in a document may have no heading if content precedes
    the first heading.

    Args:
        section_index: 0-based position in document order.
        heading: The heading that opened this section, or None for the
            pre-first-heading content block.
        span_ordinals: Ordinals of all spans that fall inside this section,
            in document order.
        start_char: Inclusive start offset of the section in the raw text.
        end_char: Exclusive end offset of the section in the raw text.
        anchor: Source anchor for this section.
    """

    section_index: int
    heading: Optional[Heading]
    span_ordinals: list[int]
    start_char: int
    end_char: int
    anchor: SectionAnchor


@dataclass
class Scene:
    """A narrative scene, bounded by scene-break markers or document edges.

    Scenes are derived from SceneBreak spans, not emitted as raw spans.

    Args:
        scene_index: 0-based position in document order.
        section_index: Index of the section this scene falls within.
        span_ordinals: Ordinals of all spans that fall inside this scene.
        start_char: Inclusive start offset in the raw text.
        end_char: Exclusive end offset in the raw text.
    """

    scene_index: int
    section_index: int
    span_ordinals: list[int]
    start_char: int
    end_char: int


@dataclass
class ParsedMarkdownDocument:
    """Output of the Markdown parser stage.

    Contains separate lists for each span type. To iterate all spans in
    document order, sort across all three lists by span_ordinal.

    Args:
        path: Document path, used as the basis for all anchors in this doc.
        raw_text: The original unmodified document text.
        headings: All heading spans in document order.
        paragraphs: All paragraph spans in document order.
        scene_breaks: All scene-break spans in document order.
        sections: All sections in document order.
        scenes: All scenes in document order.
    """

    path: str
    raw_text: str
    headings: list[Heading]
    paragraphs: list[Paragraph]
    scene_breaks: list[SceneBreak]
    sections: list[Section]
    scenes: list[Scene]


# ---------------------------------------------------------------------------
# Structured-record experiment output (structured_records/*.py)
# ---------------------------------------------------------------------------

class DocumentType(Enum):
    """Corpus-file classification before record-family segmentation."""

    MANUSCRIPT = "manuscript"
    VIGNETTE = "vignette"
    STORY_PLANNING = "story_planning"
    WORLD_CONTEXT = "world_context"
    LOCATION = "location"
    CHARACTER_BACKGROUND = "character_background"
    UNKNOWN = "unknown"


class DocumentStatus(Enum):
    """Source authority and canon-treatment metadata for one corpus file."""

    PRIMARY_CANON = "primary_canon"
    HISTORICAL = "historical"
    LEGENDARY = "legendary"
    DRAFT_UNKNOWN = "draft_unknown"
    APOCRYPHAL = "apocryphal"


@dataclass(frozen=True)
class DocumentMetadata:
    """Resolved metadata for one corpus file.

    Args:
        document_path: Source document path.
        document_type: Corpus-file document type.
        document_status: Source authority and canon-treatment status.
        status_source: Source that supplied the final status.
        status_hints: Non-authoritative hints discovered during metadata
            resolution.
        metadata_conflicts: Reviewable metadata conflicts.
    """

    document_path: str
    document_type: DocumentType
    document_status: DocumentStatus
    status_source: str
    status_hints: list[str] = field(default_factory=list)
    metadata_conflicts: list[str] = field(default_factory=list)


class StructuredRecordType(Enum):
    """High-level family for one segmented non-manuscript record."""

    REFERENCE_SECTION = "reference_section"
    DOSSIER_ENTRY = "dossier_entry"
    OUTLINE_BEAT = "outline_beat"
    LOOSE_RECORD = "loose_record"


class StructuredFieldLineType(Enum):
    """Shallow field classification inside one structured record."""

    LABEL_VALUE = "label_value"
    STANDALONE_SUBHEAD = "standalone_subhead"
    BULLET = "bullet"
    PROSE = "prose"


class StructuredEntitySource(Enum):
    """How one structured entity mention was discovered."""

    SUBJECT_HEADER = "subject_header"
    RELATIONSHIP_KEY = "relationship_key"
    RANK_TEXT = "rank_text"
    INVENTORY_MATCH = "inventory_match"


class ClaimKind(Enum):
    """Coarse kind for one retrieval-shaped claim unit."""

    FACT = "fact"
    RELATION = "relation"
    ALIAS = "alias"
    EVENT = "event"


class RetrievalResultLevel(Enum):
    """How structured one retrieval result is."""

    CLAIM_UNIT = "claim_unit"
    SOURCE_RECORD = "source_record"
    RAW_PASSAGE = "raw_passage"


class ClaimReviewState(Enum):
    """Review state for a claim-like output."""

    UNREVIEWED = "unreviewed"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"


class ClaimProposalState(Enum):
    """Authority state for a claim-like output."""

    DETERMINISTIC_PROPOSAL = "deterministic_proposal"
    LLM_PROPOSAL = "llm_proposal"
    CANONICAL = "canonical"


@dataclass
class StructuredRecord:
    """One deterministic record-sized unit cut from a non-manuscript document.

    Args:
        record_id: Stable identifier for the segmented unit.
        document_path: Source document path.
        record_type: Dominant structural family for the unit.
        anchor: Source anchor for the segmented record boundary.
        start_char: Inclusive start offset in the source text.
        end_char: Exclusive end offset in the source text.
        heading_text: Heading or header text that opened the record, when any.
        label_text: Alternate label text when the record is not heading-led.
        raw_text: Full raw text of the record.
        source_span_ordinals: Shared span-model ordinals that fall inside the
            record boundary.
        structural_flags: Deterministic structure cues preserved for later
            review and prompt construction.
        parent_heading: Nearest higher-level heading for banner context.
        ordinal_within_document: Stable document-order position of this record.
        field_lines: Raw non-heading lines kept for seed extraction.
        suspected_subject_line: Header line most likely to name the dossier
            subject, when one exists.
    """

    record_id: str
    document_path: str
    record_type: StructuredRecordType
    anchor: SpanAnchor
    start_char: int
    end_char: int
    heading_text: str = ""
    label_text: str = ""
    raw_text: str = ""
    source_span_ordinals: list[int] = field(default_factory=list)
    structural_flags: list[str] = field(default_factory=list)
    parent_heading: str = ""
    ordinal_within_document: int = 0
    field_lines: list[str] = field(default_factory=list)
    suspected_subject_line: str = ""


@dataclass
class StructuredFieldLine:
    """A shallow field-like line recovered from a structured record.

    Args:
        line_index: 0-based line index within the record body.
        line_type: Deterministic structural classification for the line.
        raw_text: Original line text.
        label: Left-side label for label-value lines.
        value: Right-side value for label-value lines.
    """

    line_index: int
    line_type: StructuredFieldLineType
    raw_text: str
    label: str = ""
    value: str = ""


@dataclass(frozen=True)
class StructuredEntityMention:
    """One entity mention recovered from structured-document signals.

    Args:
        name: Surface entity text.
        normalized_name: Canonical lowercase form used for matching.
        source: Structural extraction source.
        anchor: Source span anchor.
        record_id: Parent structured record identifier.
        document_path: Source document path.
        source_label: Local structural context label.
    """

    name: str
    normalized_name: str
    source: StructuredEntitySource
    anchor: SpanAnchor
    record_id: str
    document_path: str
    source_label: str = ""


@dataclass
class StructuredEntityInventory:
    """Cross-record structured entity inventory for one document.

    Args:
        mentions: Mentions in deterministic document order.
        names: Unique normalized names in this inventory.
        mentions_by_record: Mentions grouped by parent record id.
        records_by_name: Record ids grouped by normalized entity name.
    """

    mentions: list[StructuredEntityMention] = field(default_factory=list)
    names: frozenset[str] = field(default_factory=frozenset)
    mentions_by_record: dict[str, list[StructuredEntityMention]] = field(default_factory=dict)
    records_by_name: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DeterministicGuess:
    """A non-final deterministic semantic guess with preserved alternatives.

    Args:
        guess_type: Short guess family such as ``subject``.
        primary_guess: Current best deterministic guess.
        alternative_guesses: Other plausible deterministic alternatives.
        reason: Short explanation of why the guess exists.
        supporting_anchor: Source anchor for the evidence that produced it.
        not_final: Always True for this experiment's deterministic handoff.
    """

    guess_type: str
    primary_guess: str
    alternative_guesses: list[str]
    reason: str
    supporting_anchor: SpanAnchor
    not_final: bool = True


@dataclass
class DeterministicFactCandidate:
    """A shallow fact-like candidate preserved from one structured record.

    Args:
        label: Deterministic field label or source kind.
        value: Raw value text preserved from the record.
        reason: Why this was treated as a fact-like candidate.
        supporting_anchor: Source anchor for the record that contained it.
        line_index: Line index within the record body when available.
    """

    label: str
    value: str
    reason: str
    supporting_anchor: SpanAnchor
    line_index: int = -1


@dataclass
class ClaimEvidence:
    """Evidence span and quote supporting one retrieval-shaped claim.

    Args:
        anchor: Source anchor for the evidence.
        quote: Exact or best available quote for the evidence.
        source_snippet: Wider readable context when available.
        evidence_role: Why this evidence is attached to the claim.
    """

    anchor: SpanAnchor
    quote: str
    source_snippet: str = ""
    evidence_role: str = "primary"


@dataclass
class ClaimGroup:
    """Local grouping for related claim units from one source record.

    Args:
        claim_group_id: Stable local group identifier.
        source_record_id: Structured record that owns the group.
        group_kind: Coarse local group kind.
        group_label: Display label for the group.
        group_summary: Optional later summary of the group.
        primary_evidence_id: Stable id of the group's primary evidence.
    """

    claim_group_id: str
    source_record_id: str
    group_kind: str
    group_label: str
    group_summary: str = ""
    primary_evidence_id: str = ""


@dataclass
class ClaimUnit:
    """One atomic retrieval-shaped claim derived from extraction output.

    Args:
        claim_id: Stable identifier for this claim proposal.
        claim_kind: Coarse claim kind.
        primary_subject_guess: Best available subject guess, if any.
        alternate_subject_candidates: Other plausible subject strings.
        claim_label: Field or relation label.
        claim_value: Cleaned retrieval-friendly claim value.
        readable_summary: Human-readable summary for UI or chat packing.
        raw_claim_payload: Raw extraction payload that produced the claim.
        source_record_id: Structured record identifier.
        source_document_path: Source document path.
        document_type: Corpus-file document type.
        source_family: Source record family.
        source_status: Document status metadata, when available.
        source_authority: Source authority label from the prompt packet.
        source_authority_weight: Source authority weight after status
            metadata is applied.
        primary_evidence: Direct evidence supporting the claim.
        supporting_evidence: Additional supporting evidence.
        retrieval_channel_tags: Retrieval channels this claim can support.
        retrieval_reasons: Reasons this claim exists or may be retrieved.
        primary_retrieval_reason: Main reason to show in UI.
        review_state: Review state for the claim proposal.
        proposal_state: Whether this is deterministic, LLM, or canonical.
        claim_group: Local source-record group, when present.
        neighbor_claim_ids: Other claim ids in the same local group.
        result_level: Retrieval result level for this object.
        structure_label: Human-readable structure label.
        structure_quality: Internal structure-quality score.
    """

    claim_id: str
    claim_kind: ClaimKind
    primary_subject_guess: str
    alternate_subject_candidates: list[str]
    claim_label: str
    claim_value: str
    readable_summary: str
    raw_claim_payload: dict[str, Any]
    source_record_id: str
    source_document_path: str
    document_type: DocumentType
    source_family: str
    source_status: DocumentStatus
    source_authority: str
    source_authority_weight: float
    primary_evidence: ClaimEvidence
    supporting_evidence: list[ClaimEvidence]
    retrieval_channel_tags: list[str]
    retrieval_reasons: list[str]
    primary_retrieval_reason: str
    review_state: ClaimReviewState
    proposal_state: ClaimProposalState
    claim_group: Optional[ClaimGroup] = None
    neighbor_claim_ids: list[str] = field(default_factory=list)
    result_level: RetrievalResultLevel = RetrievalResultLevel.CLAIM_UNIT
    structure_label: str = "Extracted claim"
    structure_quality: float = 0.0


@dataclass
class DeterministicSeedBundle:
    """The deterministic packet later sent alongside raw record text.

    Args:
        record_id: Structured record identifier this seed belongs to.
        header_line: Raw header line preserved from the record.
        suspected_subject_guess: Non-final subject guess when available.
        candidate_rank_texts: Header-derived titles, ranks, or role phrases.
        field_lines: Shallow grouped field lines from the record body.
        entity_candidates: Structured entity mentions treated as deterministic
            hints only.
        reference_candidates: Reserved deferred references. This currently
            remains empty on the reference extraction path.
        known_canon_matches: Deterministic known-canon matches surfaced from
            the same local record orbit.
        structural_flags: Record-level structure cues preserved for later use.
    """

    record_id: str
    header_line: str
    suspected_subject_guess: Optional[DeterministicGuess]
    candidate_rank_texts: list[str]
    field_lines: list[StructuredFieldLine]
    entity_candidates: list["StructuredEntityMention"]
    reference_candidates: list["ReferenceCandidate"]
    known_canon_matches: list[str]
    structural_flags: list[str]


@dataclass
class RecordReviewBundle:
    """Deterministic review packet for one structured record.

    Args:
        record_id: Structured record identifier.
        record_type: Record family for this bundle.
        document_type: Corpus-file document type.
        document_status: Source authority and canon-treatment status.
        document_status_source: Source that supplied the status value.
        document_status_hints: Non-authoritative metadata hints.
        metadata_conflicts: Reviewable metadata conflicts.
        document_path: Source document path.
        raw_text: Raw record text.
        deterministic_seed_bundle: Deterministic structure and candidate packet.
        deterministic_subject_guess: Best non-final subject guess.
        deterministic_fact_candidates: Shallow fact-like candidates from the
            record.
        open_questions: Reserved review questions attached to the record.
    """

    record_id: str
    record_type: StructuredRecordType
    document_type: DocumentType
    document_status: DocumentStatus
    document_status_source: str
    document_status_hints: list[str]
    metadata_conflicts: list[str]
    document_path: str
    raw_text: str
    deterministic_seed_bundle: DeterministicSeedBundle
    deterministic_subject_guess: Optional[DeterministicGuess]
    deterministic_fact_candidates: list[DeterministicFactCandidate]
    open_questions: list[str] = field(default_factory=list)


class LLMTaskFamily(Enum):
    """LLM task families for targeted verification passes."""

    MANUSCRIPT_SUPPRESSION_RESCUE = "manuscript_suppression_rescue"


@dataclass(frozen=True)
class LLMTaskEvidenceItem:
    """Bounded evidence attached to an LLM task packet.

    Args:
        evidence_id: Stable evidence identifier.
        document_path: Source document path for this evidence.
        source_anchor: Exact source anchor for provenance and traceability.
        quote: Mention or quote text anchored by `source_anchor`.
        context_before: Left context text near the quote.
        context_after: Right context text near the quote.
        source_object_id: Parent task source object identifier.
        visibility_bucket: Deterministic evidence bucket label.
        suppression_reason: Optional suppression reason when evidence comes
            from review-only or suppressed sources.
        confidence_score: Optional deterministic confidence score.
        evidence_metadata: Optional structured metadata for retrieval-time
            materialization, such as scene references and scene excerpts.
    """

    evidence_id: str
    document_path: str
    source_anchor: SpanAnchor
    quote: str
    context_before: str
    context_after: str
    source_object_id: str
    visibility_bucket: str
    suppression_reason: str = ""
    confidence_score: float | None = None
    evidence_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMTaskPacket:
    """Shared structured LLM task packet.

    Provider runners transform this packet into model-specific messages later.
    """

    task_id: str
    task_family: LLMTaskFamily
    schema_id: str
    source_bundle_kind: str
    source_object_kind: str
    source_object_id: str
    source_document_paths: list[str]
    document_type: DocumentType
    document_status: DocumentStatus
    source_authority: str
    source_authority_weight: float
    task_goal: str
    task_constraints: list[str]
    evidence_payload: list[LLMTaskEvidenceItem]
    selection_reason: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMTaskSelectionDiagnostic:
    """Selection diagnostic for LLM task packet generation."""

    source_bundle_kind: str
    source_object_kind: str
    source_object_id: str
    document_path: str
    task_family: LLMTaskFamily
    selected: bool
    reason: str
    evidence_counts: dict[str, int] = field(default_factory=dict)


class LLMTaskResultStatus(Enum):
    """Execution status for one shared LLM task result."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class LLMTaskResult:
    """Provider execution result for one shared LLM task packet."""

    task_id: str
    task_family: LLMTaskFamily
    schema_id: str
    status: LLMTaskResultStatus
    model: str
    provider: str
    response_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class StructuredDocumentDiagnostics:
    """Structural summary for one non-manuscript experiment run.

    Args:
        document_path: Source document path.
        document_type: Corpus-file document type.
        document_status: Source authority and canon-treatment status.
        document_status_source: Source that supplied the status value.
        document_status_hints: Non-authoritative metadata hints.
        metadata_conflicts: Reviewable metadata conflicts.
        heading_count: Total detected heading spans in the parsed document.
        candidate_record_counts: Counts by segmented record family.
        sample_heading_texts: Small sample of opening heading texts.
        reason_no_review_bundles: Why no review bundles were built, if any.
    """

    document_path: str
    document_type: DocumentType
    document_status: DocumentStatus
    document_status_source: str
    document_status_hints: list[str]
    metadata_conflicts: list[str]
    heading_count: int
    candidate_record_counts: dict[str, int]
    sample_heading_texts: list[str]
    reason_no_review_bundles: str = ""


# ---------------------------------------------------------------------------
# Preprocessing stage output (preprocessing.py)
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A single token with its source offsets preserved through normalization.

    text is the normalized form used for matching. raw_text is the original
    form from the document. Offsets refer to positions in the raw document
    text so they remain valid for anchor construction.

    Args:
        text: Normalized token text (smart quotes converted, etc.).
        raw_text: Original token text before normalization.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        span_ordinal: Ordinal of the span this token came from.
    """

    text: str
    raw_text: str
    start_char: int
    end_char: int
    span_ordinal: int


@dataclass
class Sentence:
    """A sentence boundary detected within a span.

    Args:
        tokens: Tokens that make up this sentence.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        span_ordinal: Ordinal of the span this sentence came from.
    """

    tokens: list[Token]
    start_char: int
    end_char: int
    span_ordinal: int


@dataclass
class QuoteSpan:
    """A detected quoted passage.

    The offsets cover the entire quoted expression including the quotation
    marks. inner_text is the content between the marks.

    Args:
        inner_text: Text between the opening and closing quotation marks.
        start_char: Inclusive start offset covering the opening mark.
        end_char: Exclusive end offset past the closing mark.
        span_ordinal: Ordinal of the span this quote came from.
        anchor: Source anchor for this quote.
    """

    inner_text: str
    start_char: int
    end_char: int
    span_ordinal: int
    anchor: SpanAnchor


class StructuralMarkerKind(Enum):
    """Classification of a structural marker span."""

    HEADING = "heading"
    LIST_ITEM = "list_item"
    SCENE_BREAK = "scene_break"


@dataclass
class StructuralMarker:
    """A structural element tagged during preprocessing.

    Structural markers let harvesters treat headings, list items, and scene
    breaks differently from prose paragraphs without re-parsing the document.

    Args:
        kind: The structural kind of this marker.
        text: The raw text of the element.
        start_char: Inclusive start offset in the raw document text.
        end_char: Exclusive end offset in the raw document text.
        span_ordinal: Ordinal of the span this marker came from.
    """

    kind: StructuralMarkerKind
    text: str
    start_char: int
    end_char: int
    span_ordinal: int


@dataclass
class PreprocessedDocument:
    """Output of the preprocessing stage.

    The source ParsedMarkdownDocument is carried through so all subsequent
    stages have access to raw text and span structure.

    Args:
        source: The parsed document this was produced from.
        sentences: All detected sentences across all spans, in document order.
        quote_spans: All detected quoted passages, in document order.
        structural_markers: All structural markers, in document order.
        tokens_by_span: Map from span_ordinal to the tokens in that span.
    """

    source: ParsedMarkdownDocument
    sentences: list[Sentence]
    quote_spans: list[QuoteSpan]
    structural_markers: list[StructuralMarker]
    tokens_by_span: dict[int, list[Token]]


# ---------------------------------------------------------------------------
# Harvesting stage output (harvesting/*.py)
# ---------------------------------------------------------------------------

@dataclass
class MentionCandidate:
    """A candidate entity mention extracted from a span.

    Args:
        surface: Exact surface form as it appears in the document.
        normalized: Lowercased, possessive-stripped form used as the
            clustering key.
        anchor: Source anchor for this mention.
        has_title_prefix: True if the surface form starts with a recognised
            title or honorific.
        has_possessive: True if the surface form ends with 's or s'.
        has_location_context: True if the token immediately preceding this
            candidate in its span is a locative preposition (e.g. "in", "at",
            "from"). Used to classify bare-cap clusters as PLACE.
        rule_source: Label of the harvesting rule that produced this candidate.
        candidate_id: Stable ID derived from anchor path, span_ordinal, and
            surface. Use stable_hash_id to construct it.
    """

    surface: str
    normalized: str
    anchor: SpanAnchor
    has_title_prefix: bool
    has_possessive: bool
    has_location_context: bool
    rule_source: str
    candidate_id: str


@dataclass
class StructuredFieldCandidate:
    """A labeled field extracted from a structured document.

    Examples: "Alias: The Quiet One", "Role: Navigator", "Faction: The Fleet".

    Args:
        label: The field label (text before the colon or separator).
        value: The field value (text after the colon or separator).
        anchor: Source anchor for this field.
        candidate_id: Stable ID derived from anchor path, span_ordinal, and label.
    """

    label: str
    value: str
    anchor: SpanAnchor
    candidate_id: str


@dataclass
class DefinitionCandidate:
    """A definition-like term extracted from a reference or taxonomy document.

    Examples: glossary entries, acronym expansions, world-rule statements.

    Args:
        term: The term being defined.
        definition_text: The definition or explanation of the term.
        anchor: Source anchor for this definition.
        candidate_id: Stable ID derived from anchor path, span_ordinal, and term.
    """

    term: str
    definition_text: str
    anchor: SpanAnchor
    candidate_id: str


@dataclass
class SectionSummarySeed:
    """Seed material for a section-level extractive summary.

    Contains the most informative sentences from a section rather than
    attempting to generate a summary. Later scoring stages can weight these
    sentences by TF-IDF signal.

    Args:
        anchor: Source anchor for the section this seed covers.
        heading_text: Text of the section heading, or None if the section
            has no heading.
        key_sentences: The most surface-informative sentences from the section,
            in document order.
        candidate_id: Stable ID derived from anchor path and section_index.
    """

    anchor: SectionAnchor
    heading_text: Optional[str]
    key_sentences: list[str]
    candidate_id: str


# ---------------------------------------------------------------------------
# Clustering stage output (clustering/*.py)
# ---------------------------------------------------------------------------

@dataclass
class MentionCluster:
    """A cluster of MentionCandidate records grouped by normalized surface form.

    A cluster represents all occurrences of what is likely the same entity
    across the document. Surface forms may differ (titled vs. bare, possessive
    vs. plain) but share the same normalized key.

    Args:
        normalized_key: The shared normalized form used to group these
            candidates (lowercase, possessive-stripped).
        surface_forms: All distinct surface forms seen in this cluster.
        anchors: Source anchors for every individual mention in the cluster.
        occurrence_count: Total number of mention occurrences across the cluster.
        title_support_count: Number of mentions in the cluster with a title
            prefix.
        possessive_support_count: Number of mentions in possessive form.
        location_support_count: Number of mentions that appeared immediately
            after a locative preposition, indicating likely place context.
        linked_fields: Structured field candidates that reference this cluster's
            normalized key.
        linked_definitions: Definition candidates whose term matches this cluster.
        linked_seeds: Section summary seeds in sections where this cluster appears.
        cluster_id: Stable ID derived from normalized_key and document path.
    """

    normalized_key: str
    surface_forms: list[str]
    anchors: list[SpanAnchor]
    occurrence_count: int
    title_support_count: int
    possessive_support_count: int
    location_support_count: int
    linked_fields: list[StructuredFieldCandidate]
    linked_definitions: list[DefinitionCandidate]
    linked_seeds: list[SectionSummarySeed]
    cluster_id: str

    @property
    def has_title_support(self) -> bool:
        """Return True when at least one mention carries a title prefix."""
        return self.title_support_count > 0

    @property
    def has_possessive_support(self) -> bool:
        """Return True when at least one mention appears in possessive form."""
        return self.possessive_support_count > 0

    @property
    def has_location_support(self) -> bool:
        """Return True when at least one mention has locative context support."""
        return self.location_support_count > 0


# ---------------------------------------------------------------------------
# Lexicon stage output (lexicon/*.py)
# ---------------------------------------------------------------------------

class LexiconCategory(Enum):
    """Broad category assigned to a bootstrapped lexicon entry.

    UNRESOLVED is used when the evidence is strong enough to induct the entry
    but the category cannot be determined deterministically.
    """

    CHARACTER = "character"
    GROUP = "group"
    PLACE = "place"
    OBJECT = "object"
    EVENT = "event"
    CONCEPT = "concept"
    UNRESOLVED = "unresolved"


@dataclass
class BootstrappedLexiconEntry:
    """An entry in the per-document bootstrapped lexicon.

    Entries are induced from clustered evidence and compiled into an
    aho-corasick automaton for second-pass phrase matching. They are
    extraction infrastructure, not approved canon memory.

    Args:
        phrase: The canonical phrase form used for matching.
        normalized_phrase: Lowercased form of the phrase.
        category: Best-effort category classification.
        anchors: Source anchors for all evidence that supports this entry.
        occurrence_count: Total evidence occurrences that supported induction.
        archetypes_seen: Archetype labels of documents where this phrase
            appeared (e.g. "manuscript", "reference").
        rule_sources: Labels of the harvesting rules that produced the
            supporting evidence.
        induction_pass: The convergence pass (0-based) during which this
            entry was first inducted.
        entry_id: Stable ID derived from phrase and document path.
    """

    phrase: str
    normalized_phrase: str
    category: LexiconCategory
    anchors: list[SpanAnchor]
    occurrence_count: int
    archetypes_seen: list[str]
    rule_sources: list[str]
    induction_pass: int
    entry_id: str


# ---------------------------------------------------------------------------
# Promotion stage output (promotion/*.py)
# ---------------------------------------------------------------------------

class SuppressReason(Enum):
    """Why a cluster was suppressed during promotion.

    These reasons must correspond to structural or linguistic rules, not
    project-specific word lists derived from example logs.
    """

    STOPWORD = "stopword"
    SENTENCE_INITIAL_SINGLETON = "sentence_initial_singleton"
    BULLET_START_SINGLETON = "bullet_start_singleton"
    FIELD_LABEL_POSITION = "field_label_position"
    HEADING_ONLY_SINGLETON = "heading_only_singleton"
    DIALOGUE_INTERNAL = "dialogue_internal"
    GENERIC_LEXICAL_NOISE = "generic_lexical_noise"
    GENERIC_ACTION_NOUN_NOISE = "generic_action_noun_noise"
    COMPONENT_OVERLAP_NOISE = "component_overlap_noise"
    LOW_ENTITYHOOD = "low_entityhood"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class ConfidenceSignals:
    """Deterministic signals used to compute a cluster's confidence score.

    All signals are computed from the evidence without any model calls.

    Args:
        rule_tier: Highest rule tier seen in the cluster. 3 = seed lexicon
            match, 2 = titled pattern, 1 = capitalization only.
        has_title: True if at least one mention carries a title prefix.
        possessive_count: Number of possessive-form mentions in the cluster.
        attribution_count: Number of times the cluster surface appeared near
            a speech verb adjacent to a quote span.
        scene_count: Number of distinct scenes in which the cluster appears.
        tfidf_score: TF-IDF specificity score for the normalized key.
    """

    rule_tier: int
    has_title: bool
    possessive_count: int
    attribution_count: int
    scene_count: int
    tfidf_score: float


@dataclass
class PromotedCandidate:
    """A cluster that crossed the promotion threshold.

    Args:
        cluster: The underlying mention cluster.
        confidence_score: Graded score in [0.0, 1.0]. Higher means stronger
            evidence across all deterministic signals.
        signals: The individual signal values that produced the score.
        anchor: Document-level anchor for this record.
    """

    cluster: MentionCluster
    confidence_score: float
    signals: ConfidenceSignals
    anchor: DocumentAnchor


@dataclass
class ReviewOnlyCandidate:
    """A cluster with enough evidence to surface but not enough to promote.

    Args:
        cluster: The underlying mention cluster.
        confidence_score: Graded score in [0.0, 1.0].
        reason: Human-readable explanation of why this cluster was held for
            review rather than promoted or suppressed.
    """

    cluster: MentionCluster
    confidence_score: float
    reason: str


@dataclass
class SuppressedCandidate:
    """A cluster rejected by a structural suppression rule.

    Args:
        cluster: The underlying mention cluster.
        reason: The suppression rule that eliminated this cluster.
        detail: Optional extra context about why this specific cluster
            triggered the rule.
    """

    cluster: MentionCluster
    reason: SuppressReason
    detail: str


@dataclass
class SuppressedEvidence:
    """A suppressed cluster retained as secondary semantic evidence.

    Suppression controls foreground presentation, not deletion from later
    semantic review. This record preserves the suppressed cluster together
    with the rule that hid it so later stages can inspect weak evidence in
    the orbit of a stronger local container.

    Args:
        document_anchor: Source document for the suppressed evidence.
        normalized_key: Document-local normalized key for the suppressed item.
        surface_forms: Distinct observed surfaces for the suppressed cluster.
        winning_category: Document-local category hint for the cluster.
        confidence_score: Deterministic promotion score before suppression.
        reason: The structural rule that caused suppression.
        detail: Human-readable explanation of the suppression.
        anchors: Mention anchors contributing to the suppressed evidence.
    """

    document_anchor: DocumentAnchor
    normalized_key: str
    surface_forms: list[str]
    winning_category: LexiconCategory
    confidence_score: float
    reason: SuppressReason
    detail: str
    anchors: list[SpanAnchor]


@dataclass
class EvidenceWindow:
    """An entity-centric context slice intended for the retrieval interface.

    A window captures a single occurrence of an entity together with enough
    surrounding context for retrieval without requiring the full document.

    Args:
        entity_key: normalized_key of the cluster this window is for.
        anchor: Source anchor identifying the exact span occurrence.
        context_before: Text immediately preceding the entity mention.
        context_after: Text immediately following the entity mention.
        is_first_introduction: True if this is the earliest anchor in the
            cluster, which typically has the richest surrounding context.
        has_attribution: True if this occurrence is adjacent to a speech verb
            and quote span.
        speaker: Normalized key of the cluster attributed as the speaker, if
            attribution was detected.
    """

    entity_key: str
    anchor: SpanAnchor
    context_before: str
    context_after: str
    is_first_introduction: bool
    has_attribution: bool
    speaker: Optional[str]


@dataclass
class PromotedEvidenceBundle:
    """The final output of the promotion stage for one document.

    Args:
        document_anchor: Anchor identifying the source document.
        promoted: Clusters that passed the promotion threshold.
        review_only: Clusters held for human or LLM review.
        suppressed: Clusters eliminated by structural suppression rules.
        evidence_windows: Entity-centric context slices for retrieval,
            drawn from promoted and review_only clusters.
    """

    document_anchor: DocumentAnchor
    promoted: list[PromotedCandidate]
    review_only: list[ReviewOnlyCandidate]
    suppressed: list[SuppressedCandidate]
    evidence_windows: list[EvidenceWindow]


# ---------------------------------------------------------------------------
# Corpus reconciliation stage output (reconciliation/*.py)
# ---------------------------------------------------------------------------

class DocumentEntityBucket(Enum):
    """Document-local presentation tier for one entity after promotion.

    These buckets are visibility decisions for the current deterministic
    report layer, not claims about semantic truth. Suppressed records are
    hidden from primary presentation, but they still remain available for
    later semantic review.
    """

    PROMOTED = "promoted"
    REVIEW_ONLY = "review_only"
    SUPPRESSED = "suppressed"


@dataclass
class DocumentEntityRecord:
    """Stable per-document entity summary for corpus reconciliation.

    This record keeps deterministic local extraction state as nested profiles
    so later corpus and semantic stages can inspect why a local entity
    survived, how it was scored, and what evidence supports it.

    Args:
        identity: Stable local identity and display surfaces.
        current_state: Coarse top-level deterministic state.
        source_evidence: Direct anchors and context windows for this record.
        classification_trace: Full category and entityhood reasoning trace.
        promotion_trace: Promotion/suppression routing signals.
        discourse_profile: Dialogue and sentence-position usage profile.
        support_profile: Structural support counts and linkage counts.
        lineage_profile: Local compound-family structural lineage.
    """

    identity: "DocumentEntityIdentity"
    current_state: "DocumentEntityCurrentState"
    source_evidence: "DocumentEntitySourceEvidence"
    classification_trace: "DocumentEntityClassificationTrace"
    promotion_trace: "DocumentEntityPromotionTrace"
    discourse_profile: "DocumentEntityDiscourseProfile"
    support_profile: "DocumentEntitySupportProfile"
    lineage_profile: "DocumentEntityLineageProfile"


@dataclass(frozen=True)
class DocumentEntityIdentity:
    """Stable document-local identity for one entity record.

    Args:
        record_id: Stable identifier for this local record.
        document_anchor: Source document for this record.
        normalized_key: Document-local normalized cluster key.
        surface_forms: Distinct surface forms seen for the cluster.
    """

    record_id: str
    document_anchor: DocumentAnchor
    normalized_key: str
    surface_forms: list[str]


@dataclass(frozen=True)
class DocumentEntityCurrentState:
    """Coarse top-level deterministic state for one record.

    Args:
        winning_category: Final document-local top-level category.
        resolved: Whether the document-local category was resolved.
        bucket: Document-local output bucket.
    """

    winning_category: LexiconCategory
    resolved: bool
    bucket: DocumentEntityBucket


@dataclass
class DocumentEntitySourceEvidence:
    """Direct source evidence retained for one record.

    Args:
        occurrence_count: Number of supporting mentions in the document.
        anchors: Mention anchors contributing to this record.
        evidence_windows: Entity-centric context windows for later review.
        suppressed_related_evidence: Suppressed local records attached to this
            record by overlap or containment.
    """

    occurrence_count: int
    anchors: list[SpanAnchor]
    evidence_windows: list[EvidenceWindow]
    suppressed_related_evidence: list[SuppressedEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class CategoryEvidenceTrace:
    """Record-local category evidence trace.

    Args:
        category: Category this evidence describes.
        score: Deterministic confidence score for this category.
        reasons: Human-readable reasons that raised the score.
        vetoes: Human-readable reasons that reduced confidence.
    """

    category: LexiconCategory
    score: float
    reasons: list[str]
    vetoes: list[str]


@dataclass(frozen=True)
class EntityhoodTrace:
    """Record-local entityhood trace.

    Args:
        score: Deterministic entityhood score in [0.0, 1.0].
        accepted: Whether the unresolved cluster survived entityhood checks.
        reasons: Human-readable reasons that raised entityhood confidence.
        weaknesses: Human-readable reasons that kept entityhood weak.
    """

    score: float
    accepted: bool
    reasons: list[str]
    weaknesses: list[str]


@dataclass(frozen=True)
class DocumentEntityClassificationTrace:
    """Record-local classification reasoning trace.

    Args:
        winning_score: Score for the winning category.
        runner_up_category: Next-best category, if any.
        runner_up_score: Score for the runner-up category.
        evidence_by_category: Evidence by category for direct lookup.
        entityhood: Entityhood acceptance details.
    """

    winning_score: float
    runner_up_category: LexiconCategory | None
    runner_up_score: float
    evidence_by_category: dict[LexiconCategory, CategoryEvidenceTrace]
    entityhood: EntityhoodTrace


@dataclass(frozen=True)
class DocumentEntityPromotionTrace:
    """Record-local promotion and suppression routing trace.

    Args:
        confidence_score: Document-local confidence score from promotion.
        suppression_reason: Structured suppression rule when suppressed.
        bucket_detail: Human-readable reason for review or suppression.
        rule_tier: Highest structural tier seen in the cluster.
        scene_count: Number of distinct scenes containing the cluster.
        attribution_count: Number of attribution records for the key.
        possessive_count: Number of possessive mentions in the cluster.
        tfidf_score: TF-IDF style lexical weighting from scoring.
    """

    confidence_score: float
    suppression_reason: Optional[SuppressReason]
    bucket_detail: str
    rule_tier: int
    scene_count: int
    attribution_count: int
    possessive_count: int
    tfidf_score: float


@dataclass(frozen=True)
class DocumentEntityDiscourseProfile:
    """Dialogue and sentence-position usage summary for one record.

    Args:
        in_quote_count: Number of anchors inside quote spans.
        non_quote_count: Number of anchors outside quote spans.
        quote_only: Whether all anchors appear only inside quotes.
        sentence_initial_count: Anchors starting at sentence-first token.
        sentence_initial_only: Whether all anchors are sentence-initial.
        address_like_count: Anchors that match direct-address dialogue shape.
        attributed_speaker_nearby_count: Anchors near known quote speakers.
        one_token_utterance_count: Anchors that appear as one-token utterances.
    """

    in_quote_count: int
    non_quote_count: int
    quote_only: bool
    sentence_initial_count: int
    sentence_initial_only: bool
    address_like_count: int
    attributed_speaker_nearby_count: int
    one_token_utterance_count: int


@dataclass(frozen=True)
class DocumentEntitySupportProfile:
    """Structural support and linkage counts for one record.

    Args:
        title_support_count: Mentions in this cluster with title prefixes.
        possessive_support_count: Mentions in possessive form.
        location_support_count: Mentions with local location context support.
        linked_field_count: Linked structured-field count.
        linked_definition_count: Linked definition count.
        linked_seed_count: Linked bootstrap seed count.
    """

    title_support_count: int
    possessive_support_count: int
    location_support_count: int
    linked_field_count: int
    linked_definition_count: int
    linked_seed_count: int


@dataclass(frozen=True)
class DocumentEntityLineageProfile:
    """Local compound-family structural lineage for one record.

    Args:
        compound_part_count: Token part count for this normalized key.
        fully_covered_by_longer_compound: True when all anchors are covered by
            longer accepted compounds in the same spans.
        candidate_parent_keys: Sorted longer compound candidates covering this
            record's anchors.
        covered_anchor_count: Number of anchors covered by longer compounds.
        uncovered_anchor_count: Number of anchors with independent support.
        appears_as_compound_component: True when this key can be a component.
        appears_as_compound_surface: True when this key itself is a compound.
    """

    compound_part_count: int
    fully_covered_by_longer_compound: bool
    candidate_parent_keys: list[str]
    covered_anchor_count: int
    uncovered_anchor_count: int
    appears_as_compound_component: bool
    appears_as_compound_surface: bool


@dataclass
class CorpusEntity:
    """Cross-document canonical entity candidate built from document records.

    Args:
        canonical_key: Cross-document key used to group member records.
        source_keys: Exact document keys merged into this canonical entity.
        canonical_surface_forms: Observed surface forms that directly support
            the chosen canonical key.
        absorbed_surface_forms: Observed surface forms preserved from absorbed
            aliases, compounds, or deferred variants.
        member_records: All document-local records merged into this entity.
        supporting_document_paths: Distinct document paths that support it.
        dominant_category: Best current corpus-level category.
        aggregate_confidence: Conservative corpus confidence summary.
        conflicting_categories: Resolved categories that disagree across docs.
        review_required: Whether the corpus entity needs manual review.
        reasons: Human-readable explanation of merge/review decisions.
    """

    canonical_key: str
    source_keys: list[str]
    member_records: list[DocumentEntityRecord]
    supporting_document_paths: list[str]
    dominant_category: LexiconCategory
    aggregate_confidence: float
    conflicting_categories: list[LexiconCategory]
    review_required: bool
    reasons: list[str]
    canonical_surface_forms: list[str] = field(default_factory=list)
    absorbed_surface_forms: list[str] = field(default_factory=list)


@dataclass
class CorpusReconciliationResult:
    """Output of the first corpus-level exact-key reconciliation stage."""

    canonical_entities: list[CorpusEntity]


# ---------------------------------------------------------------------------
# Semantic review stage output (semantic_review/*.py)
# ---------------------------------------------------------------------------

class ReferenceCandidateType(Enum):
    """Subtype for deferred semantic reference candidates."""

    BOUND_TITLE_ROLE = "bound_title_role"
    BARE_TITLE_ROLE = "bare_title_role"
    BOUND_RELATION_ROLE = "bound_relation_role"
    BARE_RELATION_ROLE = "bare_relation_role"


class ConflictSource(Enum):
    """High-level explanation for why a corpus entity needs conflict review."""

    COMPONENT_POLLUTION = "component_pollution"
    SURFACE_LEVEL_DISAGREEMENT = "surface_level_disagreement"


class ReviewTaskKind(Enum):
    """Kinds of semantic-review tasks emitted for human or LLM review."""

    TITLE_ROLE_ATTACHMENT = "title_role_attachment"
    RELATION_ROLE_ATTACHMENT = "relation_role_attachment"
    CATEGORY_CONFLICT = "category_conflict"


class SemanticProposalSource(Enum):
    """Evidence source for a structured semantic-review proposal."""

    LOCAL_CONTEXT = "local_context"
    ADDRESS_LOCAL_CONTEXT = "address_local_context"
    DOMINANT_OWNER = "dominant_owner"


class SemanticProposalConfidence(Enum):
    """Coarse confidence tier for a semantic-review proposal."""

    LIKELY = "likely"


@dataclass
class ReferenceCandidate:
    """A deferred semantic reference mention preserved for later review.

    Reference candidates capture fiction-important mentions that are useful
    semantic evidence but are too ambiguous to force into canonical entity
    inventory during deterministic extraction.

    Args:
        document_anchor: Source document for this reference.
        reference_type: Structural subtype for the reference mention.
        surface: Exact mention surface from the document.
        normalized: Lowercased lookup form for grouping and review.
        anchor: Exact anchor for the reference mention.
        context_before: Short left context for review displays.
        context_after: Short right context for review displays.
        in_quote: Whether the mention occurs inside quoted dialogue.
        address_like: Whether the mention looks like direct address inside a
            quote, such as "Captain, wait" or "yes, captain".
        quote_speaker_key: Deterministic speaker attribution for the enclosing
            quote, when one exists.
        linked_entity_keys: Deterministic nearby entity candidates, if any.
    """

    document_anchor: DocumentAnchor
    reference_type: ReferenceCandidateType
    surface: str
    normalized: str
    anchor: SpanAnchor
    context_before: str
    context_after: str
    in_quote: bool
    address_like: bool
    quote_speaker_key: Optional[str]
    linked_entity_keys: list[str]


@dataclass
class ReferenceCluster:
    """A grouped semantic reference candidate for later attachment review.

    Repeated title or role mentions are more useful as one clustered review
    object than as many independent mentions. This grouped form preserves the
    original anchors while also exposing recurrence and candidate target counts.

    Args:
        document_anchor: Source document for this grouped reference.
        reference_type: Structural subtype for the reference.
        normalized: Lowercased lookup form shared by the grouped mentions.
        surface_forms: Distinct surfaces observed for this reference.
        occurrence_count: Total mention count in the document.
        anchors: All exact anchors contributing to this grouped reference.
        in_quote_count: Number of grouped mentions that occurred inside quotes.
        address_like_count: Number of grouped mentions that look like direct
            address inside dialogue.
        speaker_entity_scores: Quote-speaker counts by character key for the
            grouped mentions, when quote attribution was available.
        candidate_entity_scores: Deterministic target counts by entity key.
        suppressed_related_evidence: Suppressed document-local clusters that
            occur in the same local orbit as this grouped reference, kept for
            later semantic review rather than discarded.
    """

    document_anchor: DocumentAnchor
    reference_type: ReferenceCandidateType
    normalized: str
    surface_forms: list[str]
    occurrence_count: int
    anchors: list[SpanAnchor]
    in_quote_count: int
    address_like_count: int
    speaker_entity_scores: dict[str, int]
    candidate_entity_scores: dict[str, int]
    suppressed_related_evidence: list[SuppressedEvidence] = field(default_factory=list)


@dataclass
class ConflictRecord:
    """A typed cross-category conflict surfaced for semantic review.

    Args:
        canonical_key: Corpus canonical key under review.
        source: Coarse explanation of where the disagreement came from.
        conflicting_categories: Resolved categories that disagree.
        supporting_document_paths: Documents involved in the conflict.
        reason: Human-readable explanation for reports and later review.
    """

    canonical_key: str
    source: ConflictSource
    conflicting_categories: list[LexiconCategory]
    supporting_document_paths: list[str]
    reason: str


@dataclass
class ReviewTask:
    """A review prompt derived from structured semantic evidence.

    Args:
        task_id: Stable identifier for this review task.
        kind: Review task family.
        subject_key: Primary entity or reference under review.
        prompt: Human-readable review question.
        supporting_anchor_paths: Documents contributing to the question.
        ranked_candidate_keys: Ranked plausible attachment targets when the
            task concerns a deferred reference.
        ranked_speaker_keys: Ranked quote-speaker identities contributing to
            the task, when known.
        corpus_owner_keys: Ranked recurring corpus owners for the same
            normalized title or relation, excluding speakers where applicable.
        evidence_note: Short structured note explaining why this ranking was
            preserved. This is for later semantic handoff, not final truth.
    """

    task_id: str
    kind: ReviewTaskKind
    subject_key: str
    prompt: str
    supporting_anchor_paths: list[str]
    ranked_candidate_keys: list[str] = field(default_factory=list)
    ranked_speaker_keys: list[str] = field(default_factory=list)
    corpus_owner_keys: list[str] = field(default_factory=list)
    evidence_note: str = ""


@dataclass
class SemanticProposal:
    """A structured likely attachment suggested by semantic review.

    Proposals preserve the same evidence that drives review prompts, but in a
    machine-readable shape that later human or LLM passes can sort, accept,
    reject, or compare against other evidence.

    Args:
        proposal_id: Stable identifier for the proposal.
        reference_type: Reference family that produced the proposal.
        subject_key: Normalized title or relation under review.
        document_anchor: Source document for the grouped reference.
        proposed_target_key: Canonical character key suggested by the evidence.
        source: High-level reason the proposal exists.
        confidence: Coarse proposal confidence tier.
        supporting_anchors: Exact grouped mention anchors behind the proposal.
        rationale: Human-readable explanation for reports.
    """

    proposal_id: str
    reference_type: ReferenceCandidateType
    subject_key: str
    document_anchor: DocumentAnchor
    proposed_target_key: str
    source: SemanticProposalSource
    confidence: SemanticProposalConfidence
    supporting_anchors: list[SpanAnchor]
    rationale: str


@dataclass
class CharacterSemanticSummary:
    """A character-centric semantic review summary built from corpus evidence.

    These summaries give the later semantic pass a stable per-character view of
    aliases, title usage, and conflicts without forcing it to reconstruct those
    signals from the raw corpus report.

    Args:
        canonical_key: Canonical character key being summarized.
        alias_keys: Other merged keys that refer to the same character.
        canonical_surface_forms: Observed surface forms for the canonical key.
        absorbed_surface_forms: Observed absorbed or aliased surface forms
            that still belong to this character handoff object.
        supporting_document_paths: Documents that support the canonical.
        attached_title_counts: Title or role references that uniquely point to
            this character across a grouped document-level reference cluster.
        ambiguous_title_counts: Title or role references that include this
            character among multiple plausible targets and therefore still need
            attachment review.
        attached_relation_counts: Relation-role references that uniquely point
            to this character across a grouped document-level reference cluster.
        ambiguous_relation_counts: Relation-role references that include this
            character among multiple plausible targets and therefore still need
            attachment review.
        aggregate_attribution_count: Sum of document-level dialogue
            attribution counts across the merged character records.
        conflict_sources: Any typed semantic conflicts attached to this
            character canonical.
        merge_reasons: Merge and deferral reasons preserved from corpus
            reconciliation so later semantic review can see why aliases were
            folded into this canonical.
    """

    canonical_key: str
    alias_keys: list[str]
    supporting_document_paths: list[str]
    attached_title_counts: dict[str, int]
    ambiguous_title_counts: dict[str, int]
    attached_relation_counts: dict[str, int]
    ambiguous_relation_counts: dict[str, int]
    aggregate_attribution_count: int
    conflict_sources: list[ConflictSource]
    canonical_surface_forms: list[str] = field(default_factory=list)
    absorbed_surface_forms: list[str] = field(default_factory=list)
    merge_reasons: list[str] = field(default_factory=list)


@dataclass
class ManuscriptReviewBundle:
    """Persisted manuscript handoff artifact for later semantic review.

    The manuscript pipeline already has document-local entities, reference
    clusters, conflict records, and review questions. This bundle makes that
    handoff explicit in one machine-readable object so later semantic review
    can consume the same structured evidence that the human-readable report
    shows.

    Args:
        document_paths: Source documents included in the corpus run.
        entity_records: All document-local entity summaries, including
            suppressed records that still remain available as handoff evidence.
        canonical_entities: Corpus-level canonical entities after
            reconciliation.
        reference_candidates: Raw deferred semantic reference mentions.
        reference_clusters: Grouped reference-review objects.
        conflict_records: Typed cross-category conflicts preserved for review.
        character_summaries: Character-centric semantic summaries.
        review_tasks: Final question-first semantic review tasks.
    """

    document_paths: list[str]
    entity_records: list[DocumentEntityRecord]
    canonical_entities: list[CorpusEntity]
    reference_candidates: list[ReferenceCandidate]
    reference_clusters: list[ReferenceCluster]
    conflict_records: list[ConflictRecord]
    character_summaries: list[CharacterSemanticSummary]
    review_tasks: list[ReviewTask]
