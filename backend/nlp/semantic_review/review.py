"""
Semantic review helpers - derive deferred reference and conflict artifacts.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument + DocumentEntityRecord list] --> B[Scan title and relation tokens]
        B --> C[Emit ReferenceCandidate list]
        D[CorpusEntity list] --> E[Type conflict sources]
        E --> F[Emit ConflictRecord list]
        C & F --> G[Build ReviewTask list]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.harvesting.shared import RELATION_ROLE_NOUNS, TITLE_PREFIXES, stable_hash_id
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    ConflictSource,
    CorpusEntity,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    LexiconCategory,
    PreprocessedDocument,
    ReferenceCandidate,
    ReferenceCluster,
    ReferenceCandidateType,
    ReviewTask,
    ReviewTaskKind,
    SpanAnchor,
)

_TITLE_PREFIXES_NORMALIZED = frozenset(title.lower() for title in TITLE_PREFIXES)
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


def _find_enclosing_quote_range(
    token,
    ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the quote range that fully contains a token span, if any."""
    for start, end in ranges:
        if start <= token.start_char and token.end_char <= end:
            return (start, end)
    return None


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


def _is_word_like_token(token_text: str) -> bool:
    """Return True when a token behaves like lexical content rather than punctuation."""
    return any(character.isalpha() for character in token_text)


def _is_address_like_reference(sentence, token_index: int, quote_ranges: list[tuple[int, int]]) -> bool:
    """Return True when a reference token behaves like direct address in dialogue.

    This is intentionally conservative. It only marks common vocative
    patterns such as "Captain, wait." and "Yes, captain."
    """
    token = sentence.tokens[token_index]
    quote_range = _find_enclosing_quote_range(token, quote_ranges)
    if quote_range is None:
        return False

    quote_token_indexes = [
        index
        for index, quote_token in enumerate(sentence.tokens)
        if quote_range[0] <= quote_token.start_char and quote_token.end_char <= quote_range[1]
    ]
    if token_index not in quote_token_indexes:
        return False

    relative_index = quote_token_indexes.index(token_index)
    previous_indexes = quote_token_indexes[:relative_index]
    following_indexes = quote_token_indexes[relative_index + 1:]
    previous_word_indexes = [
        index for index in previous_indexes
        if _is_word_like_token(sentence.tokens[index].text)
    ]
    following_word_indexes = [
        index for index in following_indexes
        if _is_word_like_token(sentence.tokens[index].text)
    ]
    previous_token_text = sentence.tokens[previous_indexes[-1]].text if previous_indexes else ""
    following_token_text = sentence.tokens[following_indexes[0]].text if following_indexes else ""

    return (
        (not previous_word_indexes and following_token_text in {",", "!", "?"})
        or (previous_token_text == "," and not following_word_indexes)
    )


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
        if record.bucket == DocumentEntityBucket.SUPPRESSED:
            continue
        if record.winning_category != LexiconCategory.CHARACTER:
            continue
        for anchor in record.anchors:
            if anchor.path != document_path or anchor.span_ordinal != sentence.span_ordinal:
                continue
            if sentence.start_char <= anchor.start_char < sentence.end_char:
                reference_center = (reference_anchor.start_char + reference_anchor.end_char) / 2
                anchor_center = (anchor.start_char + anchor.end_char) / 2
                candidate_records.append((
                    int(abs(anchor_center - reference_center)),
                    record.confidence_score,
                    record.has_title_support,
                    anchor,
                    record.normalized_key,
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
        if record.bucket == DocumentEntityBucket.SUPPRESSED:
            continue
        if record.winning_category != LexiconCategory.CHARACTER:
            continue
        for mention_anchor in record.anchors:
            if mention_anchor.span_ordinal != sentence.span_ordinal:
                continue
            if mention_anchor.start_char == next_token.start_char:
                linked_keys.add(record.normalized_key)
                break

    return sorted(linked_keys)


def _rank_cluster_candidate_scores(
    candidate_entity_scores: dict[str, int],
    speaker_entity_scores: dict[str, int],
    address_like_count: int,
) -> dict[str, int]:
    """Return candidate scores in presentation order for semantic review.

    Address-like bare references often name the addressee rather than the
    speaker. When the current evidence says "this looks like direct address"
    and also knows who was speaking, the review surface should not present the
    speaker as the best target by default unless no better alternative exists.

    Args:
        candidate_entity_scores: Raw candidate target counts for a cluster.
        speaker_entity_scores: Counts of quote speakers observed for the same
            cluster.
        address_like_count: Number of address-like mentions in the cluster.

    Returns:
        A dict with stable insertion order that reflects review-time ranking.
    """
    if not candidate_entity_scores:
        return {}

    ranked_items = sorted(
        candidate_entity_scores.items(),
        key=lambda item: (
            address_like_count > 0 and item[0] in speaker_entity_scores,
            -item[1],
            -speaker_entity_scores.get(item[0], 0),
            item[0],
        ),
    )
    return dict(ranked_items)


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
        for anchor in record.anchors:
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
            address_like = _is_address_like_reference(
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
                    record.normalized_key
                    for record in span_records
                    if (
                        record.has_title_support
                        and record.winning_category == LexiconCategory.CHARACTER
                    )
                    and any(
                        mention_anchor.start_char <= token.start_char <= mention_anchor.end_char
                        for mention_anchor in record.anchors
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
        _TITLE_PREFIXES_NORMALIZED,
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


def extract_title_role_candidates(
    pre: PreprocessedDocument,
    records: list[DocumentEntityRecord],
    attribution_records: list,
) -> list[ReferenceCandidate]:
    """Backward-compatible wrapper for callers still using the old name."""
    return extract_reference_candidates(pre, records, attribution_records)


def build_conflict_records(entities: list[CorpusEntity]) -> list[ConflictRecord]:
    """Convert review-required corpus entities into typed conflict records.

    Args:
        entities: Corpus canonical entities from reconciliation.

    Returns:
        ConflictRecord entries for every review-required cross-category entity.
    """
    conflicts: list[ConflictRecord] = []
    for entity in sorted(entities, key=lambda item: item.canonical_key):
        if not entity.review_required or not entity.conflicting_categories:
            continue

        canonical_categories = {
            record.winning_category
            for record in entity.member_records
            if record.resolved
            and record.winning_category != LexiconCategory.UNRESOLVED
            and record.normalized_key == entity.canonical_key
        }
        absorbed_categories = {
            record.winning_category
            for record in entity.member_records
            if record.resolved
            and record.winning_category != LexiconCategory.UNRESOLVED
            and record.normalized_key != entity.canonical_key
        }

        if (
            (canonical_categories and absorbed_categories - canonical_categories)
            or (not canonical_categories and absorbed_categories and len(entity.source_keys) > 1)
        ):
            source = ConflictSource.COMPONENT_POLLUTION
            reason = (
                f"absorbed alias components introduce conflicting categories for "
                f"'{entity.canonical_key}'"
            )
        else:
            source = ConflictSource.SURFACE_LEVEL_DISAGREEMENT
            reason = (
                f"exact surface evidence for '{entity.canonical_key}' disagrees across documents"
            )

        conflicts.append(ConflictRecord(
            canonical_key=entity.canonical_key,
            source=source,
            conflicting_categories=entity.conflicting_categories,
            supporting_document_paths=entity.supporting_document_paths,
            reason=reason,
        ))

    return conflicts


def build_reference_clusters(references: list[ReferenceCandidate]) -> list[ReferenceCluster]:
    """Group repeated reference candidates into stable document-level clusters.

    Args:
        references: Raw reference candidates extracted from one or more
            documents.

    Returns:
        Grouped reference clusters in document and subtype order.
    """
    grouped: dict[tuple[str, str, str], list[ReferenceCandidate]] = defaultdict(list)
    bound_entity_scores: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for reference in references:
        grouped[(
            reference.document_anchor.path,
            reference.reference_type.value,
            reference.normalized,
        )].append(reference)
        if reference.reference_type in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BOUND_RELATION_ROLE,
        }:
            for key in reference.linked_entity_keys:
                bound_entity_scores[(reference.document_anchor.path, reference.normalized)][key] += 1

    clusters: list[ReferenceCluster] = []
    for (_path, _rtype, _normalized), members in sorted(grouped.items()):
        candidate_entity_scores: dict[str, int] = defaultdict(int)
        speaker_entity_scores: dict[str, int] = defaultdict(int)
        for member in members:
            for key in member.linked_entity_keys:
                candidate_entity_scores[key] += 1
            if member.quote_speaker_key is not None:
                speaker_entity_scores[member.quote_speaker_key] += 1
        if members[0].reference_type in {
            ReferenceCandidateType.BARE_TITLE_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            bound_scores = bound_entity_scores[(members[0].document_anchor.path, members[0].normalized)]
            if not candidate_entity_scores and len(bound_scores) == 1:
                only_key = next(iter(bound_scores))
                candidate_entity_scores[only_key] = len(members)
        ranked_speaker_scores = dict(sorted(
            speaker_entity_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        ranked_candidate_scores = _rank_cluster_candidate_scores(
            dict(candidate_entity_scores),
            ranked_speaker_scores,
            sum(1 for member in members if member.address_like),
        )
        clusters.append(ReferenceCluster(
            document_anchor=members[0].document_anchor,
            reference_type=members[0].reference_type,
            normalized=members[0].normalized,
            surface_forms=sorted({member.surface for member in members}),
            occurrence_count=len(members),
            anchors=sorted(
                [member.anchor for member in members],
                key=lambda anchor: (anchor.path, anchor.span_ordinal, anchor.start_char),
            ),
            in_quote_count=sum(1 for member in members if member.in_quote),
            address_like_count=sum(1 for member in members if member.address_like),
            speaker_entity_scores=ranked_speaker_scores,
            candidate_entity_scores=ranked_candidate_scores,
        ))

    return clusters


def build_character_summaries(
    entities: list[CorpusEntity],
    reference_clusters: list[ReferenceCluster],
    conflicts: list[ConflictRecord],
) -> list[CharacterSemanticSummary]:
    """Build character-centric semantic summaries from corpus review evidence.

    The deterministic pipeline already knows a lot about recurring characters,
    but that evidence is spread across entity aliases, grouped title mentions,
    and typed conflicts. This pass gathers those signals into one stable
    review object per canonical character so later semantic review can reason
    per person instead of scanning flat corpus tables.

    Args:
        entities: Corpus canonical entities from reconciliation.
        reference_clusters: Grouped semantic reference candidates.
        conflicts: Typed conflict records already built for the corpus.

    Returns:
        One summary per canonical character in stable presentation order.
    """
    character_entities = [
        entity for entity in entities
        if entity.dominant_category == LexiconCategory.CHARACTER
    ]
    character_keys_by_source: dict[str, str] = {}
    for entity in character_entities:
        for key in entity.source_keys:
            character_keys_by_source[key] = entity.canonical_key

    attached_title_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ambiguous_title_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    attached_relation_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ambiguous_relation_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cluster in reference_clusters:
        canonical_targets = sorted({
            character_keys_by_source[key]
            for key in cluster.candidate_entity_scores
            if key in character_keys_by_source
        })
        if cluster.reference_type in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BARE_TITLE_ROLE,
        }:
            attached_counts = attached_title_counts
            ambiguous_counts = ambiguous_title_counts
        elif cluster.reference_type in {
            ReferenceCandidateType.BOUND_RELATION_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            attached_counts = attached_relation_counts
            ambiguous_counts = ambiguous_relation_counts
        else:
            continue
        if len(canonical_targets) == 1:
            attached_counts[canonical_targets[0]][cluster.normalized] += cluster.occurrence_count
        elif len(canonical_targets) > 1:
            for canonical_key in canonical_targets:
                ambiguous_counts[canonical_key][cluster.normalized] += cluster.occurrence_count

    conflicts_by_key: dict[str, list[ConflictSource]] = defaultdict(list)
    for conflict in conflicts:
        conflicts_by_key[conflict.canonical_key].append(conflict.source)

    summaries: list[CharacterSemanticSummary] = []
    for entity in sorted(
        character_entities,
        key=lambda item: (-len(item.supporting_document_paths), item.canonical_key),
    ):
        summaries.append(CharacterSemanticSummary(
            canonical_key=entity.canonical_key,
            alias_keys=[key for key in entity.source_keys if key != entity.canonical_key],
            supporting_document_paths=entity.supporting_document_paths,
            attached_title_counts=dict(sorted(
                attached_title_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            ambiguous_title_counts=dict(sorted(
                ambiguous_title_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            attached_relation_counts=dict(sorted(
                attached_relation_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            ambiguous_relation_counts=dict(sorted(
                ambiguous_relation_counts[entity.canonical_key].items(),
                key=lambda item: (-item[1], item[0]),
            )),
            aggregate_attribution_count=sum(
                record.attribution_count for record in entity.member_records
            ),
            conflict_sources=sorted(
                conflicts_by_key[entity.canonical_key],
                key=lambda source: source.value,
            ),
        ))

    return summaries


def _build_title_owner_scores(
    reference_clusters: list[ReferenceCluster],
) -> dict[str, dict[str, int]]:
    """Aggregate recurring title ownership hints across the current corpus.

    Unique title attachments in one document are useful fallback evidence for
    harder address-like uses in another. This helper stays conservative by
    counting only title clusters that already point to exactly one character.

    Args:
        reference_clusters: Grouped semantic reference candidates.

    Returns:
        Mapping of normalized title to per-character evidence counts.
    """
    title_owner_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cluster in reference_clusters:
        if cluster.reference_type not in {
            ReferenceCandidateType.BOUND_TITLE_ROLE,
            ReferenceCandidateType.BARE_TITLE_ROLE,
        }:
            continue
        if len(cluster.candidate_entity_scores) != 1:
            continue
        only_key = next(iter(cluster.candidate_entity_scores))
        title_owner_scores[cluster.normalized][only_key] += cluster.occurrence_count

    return {
        normalized: dict(sorted(
            owner_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        for normalized, owner_scores in title_owner_scores.items()
    }


def _build_canonical_title_owner_scores(
    character_summaries: list[CharacterSemanticSummary],
) -> dict[str, dict[str, int]]:
    """Aggregate title ownership by canonical character key.

    Character summaries already fold aliases into one canonical person, so
    title counts derived from them are better fallback evidence than raw
    document-level keys when a review prompt needs corpus ownership hints.

    Args:
        character_summaries: Character-centric semantic summaries.

    Returns:
        Mapping of normalized title to canonical owner counts.
    """
    title_owner_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in character_summaries:
        for normalized, count in summary.attached_title_counts.items():
            title_owner_scores[normalized][summary.canonical_key] += count

    return {
        normalized: dict(sorted(
            owner_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        for normalized, owner_scores in title_owner_scores.items()
    }


def _build_canonical_relation_owner_scores(
    character_summaries: list[CharacterSemanticSummary],
) -> dict[str, dict[str, int]]:
    """Aggregate relation ownership by canonical character key."""
    relation_owner_scores: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in character_summaries:
        for normalized, count in summary.attached_relation_counts.items():
            relation_owner_scores[normalized][summary.canonical_key] += count

    return {
        normalized: dict(sorted(
            owner_scores.items(),
            key=lambda item: (-item[1], item[0]),
        ))
        for normalized, owner_scores in relation_owner_scores.items()
    }


def _build_character_canonical_map(
    character_summaries: list[CharacterSemanticSummary],
) -> dict[str, str]:
    """Map character aliases back to canonical character keys."""
    canonical_map: dict[str, str] = {}
    for summary in character_summaries:
        canonical_map[summary.canonical_key] = summary.canonical_key
        for alias_key in summary.alias_keys:
            canonical_map[alias_key] = summary.canonical_key
    return canonical_map


def _strong_owner_keys(owner_scores: dict[str, int]) -> list[str]:
    """Return only the strongest ownership candidates for review prompts.

    Corpus-level fallback should help a human or later semantic pass focus on
    plausible owners, not enumerate every weak one-off holder of a title.

    Args:
        owner_scores: Ranked owner counts for one normalized title.

    Returns:
        A short list of the strongest owner keys.
    """
    if not owner_scores:
        return []

    top_score = max(owner_scores.values())
    threshold = max(2, (top_score + 1) // 2)
    strong_keys = [
        key for key, score in owner_scores.items()
        if score >= threshold
    ]
    return strong_keys[:3]


def _dominant_owner_key(owner_scores: dict[str, int]) -> str | None:
    """Return one owner when corpus evidence clearly favors a single target.

    Args:
        owner_scores: Ranked owner counts for one normalized title.

    Returns:
        The dominant owner key when the top owner clearly separates from the
        runner-up, otherwise None.
    """
    if not owner_scores:
        return None

    ranked_items = list(owner_scores.items())
    top_key, top_score = ranked_items[0]
    if top_score < 3:
        return None
    if len(ranked_items) == 1:
        return top_key

    second_score = ranked_items[1][1]
    if top_score >= second_score * 2:
        return top_key
    return None


def _canonicalize_ranked_keys(
    keys: list[str],
    canonical_map: dict[str, str],
) -> list[str]:
    """Collapse alias keys to canonical character keys in stable order."""
    canonical_keys: list[str] = []
    seen: set[str] = set()
    for key in keys:
        canonical_key = canonical_map.get(key, key)
        if canonical_key in seen:
            continue
        seen.add(canonical_key)
        canonical_keys.append(canonical_key)
    return canonical_keys


def build_review_tasks(
    reference_clusters: list[ReferenceCluster],
    conflicts: list[ConflictRecord],
    character_summaries: list[CharacterSemanticSummary] | None = None,
) -> list[ReviewTask]:
    """Build lightweight semantic review prompts from structured evidence.

    Args:
        reference_clusters: Grouped deferred title and role references.
        conflicts: Typed cross-category conflict records.

    Returns:
        Stable review tasks for later human or LLM review.
    """
    deduped_tasks: dict[tuple[str, str, str, tuple[str, ...]], ReviewTask] = {}
    title_owner_scores = (
        _build_canonical_title_owner_scores(character_summaries)
        if character_summaries is not None
        else _build_title_owner_scores(reference_clusters)
    )
    relation_owner_scores = (
        _build_canonical_relation_owner_scores(character_summaries)
        if character_summaries is not None
        else {}
    )
    canonical_map = (
        _build_character_canonical_map(character_summaries)
        if character_summaries is not None
        else {}
    )

    for reference in reference_clusters:
        if reference.reference_type not in {
            ReferenceCandidateType.BARE_TITLE_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            continue
        ranked_entity_keys = _canonicalize_ranked_keys(
            list(reference.candidate_entity_scores.keys()),
            canonical_map,
        )
        ranked_speaker_keys = _canonicalize_ranked_keys(
            list(reference.speaker_entity_scores.keys()),
            canonical_map,
        )
        non_speaker_entity_keys = [
            key for key in ranked_entity_keys
            if key not in ranked_speaker_keys
        ]
        corpus_title_owner_keys: list[str] = []
        dominant_owner_key = None
        if reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE:
            non_speaker_owner_scores = {
                key: score
                for key, score in title_owner_scores.get(reference.normalized, {}).items()
                if key not in ranked_speaker_keys
            }
            corpus_title_owner_keys = _strong_owner_keys(non_speaker_owner_scores)
            dominant_owner_key = _dominant_owner_key(non_speaker_owner_scores)
        corpus_relation_owner_keys: list[str] = []
        dominant_relation_owner_key = None
        if reference.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE:
            relation_scores = relation_owner_scores.get(reference.normalized, {})
            non_speaker_relation_scores = {
                key: score
                for key, score in relation_scores.items()
                if key not in ranked_speaker_keys
            }
            corpus_relation_owner_keys = _strong_owner_keys(non_speaker_relation_scores)
            dominant_relation_owner_key = _dominant_owner_key(non_speaker_relation_scores)
        speaker_text = ""
        if reference.address_like_count and ranked_speaker_keys:
            speaker_text = f" spoken by {', '.join(ranked_speaker_keys)}"
        if reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE:
            kind = ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            label = "title"
        else:
            kind = ReviewTaskKind.RELATION_ROLE_ATTACHMENT
            label = "relation"
        if reference.address_like_count and ranked_speaker_keys and non_speaker_entity_keys:
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} refer to one of {', '.join(non_speaker_entity_keys)}?"
            )
        elif (
            reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE
            and reference.address_like_count
            and ranked_speaker_keys
            and dominant_owner_key is not None
        ):
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} most likely refer to {dominant_owner_key}?"
            )
        elif (
            reference.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE
            and reference.address_like_count
            and ranked_speaker_keys
            and dominant_relation_owner_key is not None
        ):
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} most likely refer to {dominant_relation_owner_key}?"
            )
        elif (
            reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE
            and reference.address_like_count
            and ranked_speaker_keys
            and corpus_title_owner_keys
        ):
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} refer to one of {', '.join(corpus_title_owner_keys)}?"
            )
        elif (
            reference.reference_type == ReferenceCandidateType.BARE_RELATION_ROLE
            and reference.address_like_count
            and ranked_speaker_keys
            and corpus_relation_owner_keys
        ):
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} refer to one of {', '.join(corpus_relation_owner_keys)}?"
            )
        elif reference.address_like_count and ranked_speaker_keys and ranked_entity_keys:
            prompt = (
                f"Does the address-like bare {label} '{reference.normalized}'"
                f"{speaker_text} refer to a recurring character other than the speaker?"
            )
        elif ranked_entity_keys:
            prompt = (
                f"Does the "
                f"{'address-like ' if reference.address_like_count else ''}"
                f"bare {label} '{reference.normalized}'{speaker_text} refer to one of "
                f"{', '.join(ranked_entity_keys)}?"
            )
        else:
            prompt = (
                f"Does the "
                f"{'address-like ' if reference.address_like_count else ''}"
                f"bare {label} '{reference.normalized}'{speaker_text} refer to a recurring "
                f"character or role in this document?"
            )
        task = ReviewTask(
            task_id=stable_hash_id(
                reference.document_anchor.path,
                reference.reference_type.value,
                reference.normalized,
            ),
            kind=kind,
            subject_key=reference.normalized,
            prompt=prompt,
            supporting_anchor_paths=[reference.document_anchor.path],
        )
        deduped_tasks[(
            task.kind.value,
            task.subject_key,
            task.prompt,
            tuple(task.supporting_anchor_paths),
        )] = task

    for conflict in conflicts:
        task = ReviewTask(
            task_id=stable_hash_id(
                conflict.canonical_key,
                ",".join(category.value for category in conflict.conflicting_categories),
            ),
            kind=ReviewTaskKind.CATEGORY_CONFLICT,
            subject_key=conflict.canonical_key,
            prompt=(
                f"Resolve the category conflict for '{conflict.canonical_key}': "
                f"{', '.join(category.value for category in conflict.conflicting_categories)}"
            ),
            supporting_anchor_paths=conflict.supporting_document_paths,
        )
        deduped_tasks[(
            task.kind.value,
            task.subject_key,
            task.prompt,
            tuple(task.supporting_anchor_paths),
        )] = task

    return sorted(
        deduped_tasks.values(),
        key=lambda task: (task.kind.value, task.subject_key, task.prompt),
    )
