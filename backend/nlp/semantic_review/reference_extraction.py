"""
Semantic reference extraction - detect deferred title and relation mentions.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument + DocumentEntityRecord list] --> B[Index quotes and span records]
        B --> C[Scan sentence tokens for title and relation lexicons]
        C --> D[Resolve local character links and quote-speaker context]
        D --> E[Emit ReferenceCandidate list]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.harvesting.shared import (
    RELATION_ROLE_NOUNS,
    TITLE_PREFIXES_LOWER,
)
from backend.nlp.discourse.address_like import is_address_like_reference
from backend.nlp.types import (
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
    PreprocessedDocument,
    ReferenceCandidate,
    ReferenceCandidateType,
    SpanAnchor,
)

_RELATION_ROLE_NOUNS_NORMALIZED = frozenset(noun.lower() for noun in RELATION_ROLE_NOUNS)
_CONTEXT_WINDOW_CHARS = 60


def _context_slice(raw_text: str, anchor: SpanAnchor) -> tuple[str, str]:
    """Return short left and right context around an anchor.

    Args:
        raw_text: Full document text.
        anchor: Exact span anchor to slice around.

    Returns:
        Two short strings: left context and right context.
    """
    left_start = max(0, anchor.start_char - _CONTEXT_WINDOW_CHARS)
    right_end = min(len(raw_text), anchor.end_char + _CONTEXT_WINDOW_CHARS)
    return raw_text[left_start:anchor.start_char], raw_text[anchor.end_char:right_end]


def _quote_ranges(pre: PreprocessedDocument) -> dict[int, list[tuple[int, int]]]:
    """Index quote ranges by span ordinal for fast containment checks."""
    ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for quote in pre.quote_spans:
        ranges[quote.span_ordinal].append((quote.start_char, quote.end_char))
    return ranges


def _quote_speakers_by_anchor(attribution_records: list) -> dict[tuple[str, int, int, int], str]:
    """Index quote speakers by exact quote anchor for semantic-reference use."""
    speakers: dict[tuple[str, int, int, int], str] = {}
    for record in attribution_records:
        key = (
            record.quote_anchor.path,
            record.quote_anchor.span_ordinal,
            record.quote_anchor.start_char,
            record.quote_anchor.end_char,
        )
        speakers[key] = record.speaker_key
    return speakers



def _find_enclosing_quote_anchor(pre: PreprocessedDocument, token) -> SpanAnchor | None:
    """Return the exact quote anchor that encloses a token span, if any."""
    for quote in pre.quote_spans:
        if (
            quote.span_ordinal == token.span_ordinal
            and quote.start_char <= token.start_char
            and token.end_char <= quote.end_char
        ):
            return quote.anchor
    return None



def _find_sentence_entities(
    sentence,
    records: list[DocumentEntityRecord],
    document_path: str,
    reference_anchor: SpanAnchor,
) -> list[str]:
    """Return plausible character keys whose anchors overlap a sentence span.

    Bare title mentions are useful semantic evidence, but nearby unresolved
    survivors are often discourse fragments or generic nouns. This helper
    therefore only surfaces character-like sentence anchors for later title
    attachment review and leaves weaker surfaces to the fallback path that
    uses explicit bound title evidence.

    Args:
        sentence: Sentence whose span will be searched.
        records: Stable document entity summaries for the same document.
        document_path: Source path for the current document.

    Returns:
        Ranked normalized keys for overlapping character records.
    """
    candidate_records: list[tuple[int, float, bool, SpanAnchor, str]] = []
    for record in records:
        if record.current_state.bucket == DocumentEntityBucket.SUPPRESSED:
            continue
        if record.current_state.winning_category != LexiconCategory.CHARACTER:
            continue
        for anchor in record.source_evidence.anchors:
            if anchor.path != document_path or anchor.span_ordinal != sentence.span_ordinal:
                continue
            if sentence.start_char <= anchor.start_char < sentence.end_char:
                reference_center = (reference_anchor.start_char + reference_anchor.end_char) / 2
                anchor_center = (anchor.start_char + anchor.end_char) / 2
                candidate_records.append((
                    int(abs(anchor_center - reference_center)),
                    record.promotion_trace.confidence_score,
                    record.support_profile.title_support_count > 0,
                    anchor,
                    record.identity.normalized_key,
                ))
                break

    candidate_records.sort(key=lambda item: (item[3].start_char, item[3].end_char, item[4]))

    overlap_groups: list[list[tuple[int, float, bool, SpanAnchor, str]]] = []
    for candidate in candidate_records:
        if not overlap_groups:
            overlap_groups.append([candidate])
            continue
        last_group = overlap_groups[-1]
        last_end = max(item[3].end_char for item in last_group)
        if candidate[3].start_char < last_end:
            last_group.append(candidate)
        else:
            overlap_groups.append([candidate])

    chosen: list[tuple[SpanAnchor, str, int, float, bool]] = []
    for group in overlap_groups:
        best = min(
            group,
            key=lambda item: (-item[1], item[0], not item[2], item[3].end_char - item[3].start_char, item[4]),
        )
        chosen.append((best[3], best[4], best[0], best[1], best[2]))

    chosen.sort(key=lambda item: (item[2], -item[3], not item[4], item[0].start_char, item[1]))

    return [key for _anchor, key, _distance, _confidence, _titled in chosen[:3]]


def _find_bound_relation_links(
    sentence,
    token_index: int,
    span_records: list[DocumentEntityRecord],
) -> list[str]:
    """Return nearby character anchors for a relation noun plus name pattern.

    Relation nouns such as "brother" and "mentor" are usually not part of
    the harvested character span themselves. They sit immediately before the
    actual name token, so bound detection needs to look forward to the next
    token's anchor rather than checking whether the relation token sits inside
    an existing mention span.

    Args:
        sentence: Sentence containing the relation token.
        token_index: Index of the relation token inside the sentence.
        span_records: Document entity records that overlap this sentence span.

    Returns:
        Sorted character keys whose mention span begins at the next token.
    """
    if token_index + 1 >= len(sentence.tokens):
        return []

    next_token = sentence.tokens[token_index + 1]
    linked_keys: set[str] = set()
    for record in span_records:
        if record.current_state.bucket == DocumentEntityBucket.SUPPRESSED:
            continue
        if record.current_state.winning_category != LexiconCategory.CHARACTER:
            continue
        for mention_anchor in record.source_evidence.anchors:
            if mention_anchor.span_ordinal != sentence.span_ordinal:
                continue
            if mention_anchor.start_char == next_token.start_char:
                linked_keys.add(record.identity.normalized_key)
                break

    return sorted(linked_keys)


def _extract_reference_candidates_for_lexicon(
    pre: PreprocessedDocument,
    records: list[DocumentEntityRecord],
    attribution_records: list,
    lexicon: frozenset[str],
    bound_type: ReferenceCandidateType,
    bare_type: ReferenceCandidateType,
    require_title_support_for_bound: bool,
) -> list[ReferenceCandidate]:
    """Extract deferred semantic references from one lexical inventory.

    The semantic-review layer preserves linguistically meaningful references
    such as titles and kinship nouns without forcing them into canonical
    entity inventory too early. Bound uses inherit stronger evidence from
    nearby named character spans, while bare uses keep only conservative
    local character candidates.

    Args:
        pre: Preprocessed document for token and quote access.
        records: Stable per-document entity summaries for this document.
        attribution_records: Dialogue attribution records for the document.
        lexicon: Lowercased words that count as semantic reference triggers.
        bound_type: Candidate subtype for bound mentions.
        bare_type: Candidate subtype for bare mentions.
        require_title_support_for_bound: Whether bound matches should only
            attach to title-supported entity records.

    Returns:
        ReferenceCandidate records in document order.
    """
    raw_text = pre.source.raw_text
    quote_ranges = _quote_ranges(pre)
    quote_speakers = _quote_speakers_by_anchor(attribution_records)
    records_by_span: dict[int, list[DocumentEntityRecord]] = defaultdict(list)
    for record in records:
        for anchor in record.source_evidence.anchors:
            records_by_span[anchor.span_ordinal].append(record)

    candidates: list[ReferenceCandidate] = []
    for sentence in pre.sentences:
        span_records = records_by_span.get(sentence.span_ordinal, [])
        for token_index, token in enumerate(sentence.tokens):
            normalized = token.text.lower().rstrip(".")
            if normalized not in lexicon:
                continue

            anchor = SpanAnchor(
                path=pre.source.path,
                span_ordinal=sentence.span_ordinal,
                start_char=token.start_char,
                end_char=token.end_char,
            )
            sentence_entity_keys = _find_sentence_entities(
                sentence,
                records,
                pre.source.path,
                anchor,
            )
            sentence_quote_ranges = quote_ranges.get(sentence.span_ordinal, [])
            inside_quotes = any(
                start <= token.start_char and token.end_char <= end
                for start, end in sentence_quote_ranges
            )
            address_like = is_address_like_reference(
                sentence,
                token_index,
                sentence_quote_ranges,
            )
            quote_anchor = _find_enclosing_quote_anchor(pre, token)
            quote_speaker_key = None
            if quote_anchor is not None:
                quote_speaker_key = quote_speakers.get((
                    quote_anchor.path,
                    quote_anchor.span_ordinal,
                    quote_anchor.start_char,
                    quote_anchor.end_char,
                ))

            if require_title_support_for_bound:
                linked_entity_keys = sorted({
                    record.identity.normalized_key
                    for record in span_records
                    if (
                        record.support_profile.title_support_count > 0
                        and record.current_state.winning_category == LexiconCategory.CHARACTER
                    )
                    and any(
                        mention_anchor.start_char <= token.start_char <= mention_anchor.end_char
                        for mention_anchor in record.source_evidence.anchors
                    )
                })
            else:
                linked_entity_keys = _find_bound_relation_links(
                    sentence,
                    token_index,
                    span_records,
                )
            reference_type = bound_type
            if not linked_entity_keys:
                reference_type = bare_type
                linked_entity_keys = [
                    key for key in sentence_entity_keys
                    if key != normalized
                ]

            context_before, context_after = _context_slice(raw_text, anchor)
            candidates.append(ReferenceCandidate(
                document_anchor=DocumentAnchor(path=pre.source.path),
                reference_type=reference_type,
                surface=token.text,
                normalized=normalized,
                anchor=anchor,
                context_before=context_before,
                context_after=context_after,
                in_quote=inside_quotes,
                address_like=address_like,
                quote_speaker_key=quote_speaker_key,
                linked_entity_keys=linked_entity_keys,
            ))

    return candidates


def extract_reference_candidates(
    pre: PreprocessedDocument,
    records: list[DocumentEntityRecord],
    attribution_records: list,
) -> list[ReferenceCandidate]:
    """Extract deferred title and relation references from document evidence.

    Titles and kinship or relation nouns are both fiction-important reference
    signals. They are preserved together here because later semantic review
    will often need to consider them side by side when deciding whether a
    scene refers to one known character, several characters, or a missing
    referent.

    Args:
        pre: Preprocessed document for token and quote access.
        records: Stable per-document entity summaries for this document.
        attribution_records: Dialogue attribution records for the document.

    Returns:
        ReferenceCandidate records in document order.
    """
    title_candidates = _extract_reference_candidates_for_lexicon(
        pre,
        records,
        attribution_records,
        TITLE_PREFIXES_LOWER,
        ReferenceCandidateType.BOUND_TITLE_ROLE,
        ReferenceCandidateType.BARE_TITLE_ROLE,
        True,
    )
    relation_candidates = _extract_reference_candidates_for_lexicon(
        pre,
        records,
        attribution_records,
        _RELATION_ROLE_NOUNS_NORMALIZED,
        ReferenceCandidateType.BOUND_RELATION_ROLE,
        ReferenceCandidateType.BARE_RELATION_ROLE,
        False,
    )
    return sorted(
        title_candidates + relation_candidates,
        key=lambda candidate: (
            candidate.anchor.path,
            candidate.anchor.span_ordinal,
            candidate.anchor.start_char,
            candidate.reference_type.value,
        ),
    )
