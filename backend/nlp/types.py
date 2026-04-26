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
from typing import Optional


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
        rule_source: Label of the harvesting rule that produced this candidate.
        candidate_id: Stable ID derived from anchor path, span_ordinal, and
            surface. Use stable_hash_id to construct it.
    """

    surface: str
    normalized: str
    anchor: SpanAnchor
    has_title_prefix: bool
    has_possessive: bool
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
        has_title_support: True if any mention in the cluster has a title prefix.
        has_possessive_support: True if any mention has a possessive form.
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
    has_title_support: bool
    has_possessive_support: bool
    linked_fields: list[StructuredFieldCandidate]
    linked_definitions: list[DefinitionCandidate]
    linked_seeds: list[SectionSummarySeed]
    cluster_id: str


# ---------------------------------------------------------------------------
# Lexicon stage output (lexicon/*.py)
# ---------------------------------------------------------------------------

class LexiconCategory(Enum):
    """Broad category assigned to a bootstrapped lexicon entry.

    UNRESOLVED is used when the evidence is strong enough to induct the entry
    but the category cannot be determined deterministically.
    """

    CHARACTER = "character"
    PLACE = "place"
    FACTION = "faction"
    ARTIFACT = "artifact"
    TERMINOLOGY = "terminology"
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
