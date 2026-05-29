"""
Per-document entity summaries for later corpus reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster list] --> B[Iterate clusters]
        C[PromotedEvidenceBundle] --> D[Bucket lookup]
        E[Promotion scores + classifications + attributions] --> B
        B & D --> F[DocumentEntityRecord list]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.classification.compound_shapes import compound_parts
from backend.nlp.classification.types import ClassificationDecision
from backend.nlp.discourse.address_like import is_address_like_reference
from backend.nlp.promotion.attribution import AttributionRecord
from backend.nlp.types import (
    CategoryEvidenceTrace,
    ConfidenceSignals,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityClassificationTrace,
    DocumentEntityCurrentState,
    DocumentEntityDiscourseProfile,
    DocumentEntityIdentity,
    DocumentEntityLineageProfile,
    DocumentEntityPromotionTrace,
    DocumentEntityRecord,
    DocumentEntitySourceEvidence,
    DocumentEntitySupportProfile,
    EntityhoodTrace,
    LexiconCategory,
    MentionCluster,
    PreprocessedDocument,
    PromotedEvidenceBundle,
    SpanAnchor,
    SuppressedEvidence,
    stable_hash_id,
)


def _anchors_overlap(left: SpanAnchor, right: SpanAnchor) -> bool:
    """Return True when two anchors overlap in the same span."""
    return (
        left.path == right.path
        and left.span_ordinal == right.span_ordinal
        and left.start_char < right.end_char
        and right.start_char < left.end_char
    )


def _anchor_contains(outer: SpanAnchor, inner: SpanAnchor) -> bool:
    """Return True when one anchor fully contains the other in the same span."""
    return (
        outer.path == inner.path
        and outer.span_ordinal == inner.span_ordinal
        and outer.start_char <= inner.start_char
        and inner.end_char <= outer.end_char
    )


def _build_suppressed_evidence(record: DocumentEntityRecord) -> SuppressedEvidence:
    """Convert a suppressed document entity record into retained evidence."""
    assert record.promotion_trace.suppression_reason is not None
    return SuppressedEvidence(
        document_anchor=record.identity.document_anchor,
        normalized_key=record.identity.normalized_key,
        surface_forms=list(record.identity.surface_forms),
        winning_category=record.current_state.winning_category,
        confidence_score=record.promotion_trace.confidence_score,
        reason=record.promotion_trace.suppression_reason,
        detail=record.promotion_trace.bucket_detail,
        anchors=list(record.source_evidence.anchors),
    )


def _attach_suppressed_evidence_to_records(
    records: list[DocumentEntityRecord],
) -> None:
    """Attach suppressed records beneath stronger overlapping local entities."""
    survivors = [
        record for record in records
        if record.current_state.bucket != DocumentEntityBucket.SUPPRESSED
    ]
    suppressed = [
        record for record in records
        if record.current_state.bucket == DocumentEntityBucket.SUPPRESSED
    ]

    for suppressed_record in suppressed:
        best_target: DocumentEntityRecord | None = None
        best_score: tuple[int, int, float, int, str] | None = None
        for target in survivors:
            containment_count = 0
            overlap_count = 0
            for suppressed_anchor in suppressed_record.source_evidence.anchors:
                for target_anchor in target.source_evidence.anchors:
                    if not _anchors_overlap(suppressed_anchor, target_anchor):
                        continue
                    overlap_count += 1
                    if _anchor_contains(target_anchor, suppressed_anchor):
                        containment_count += 1
            if overlap_count == 0:
                continue

            score = (
                containment_count,
                overlap_count,
                target.promotion_trace.confidence_score,
                target.source_evidence.occurrence_count,
                target.identity.normalized_key,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_target = target

        if best_target is None:
            continue

        best_target.source_evidence.suppressed_related_evidence.append(
            _build_suppressed_evidence(suppressed_record)
        )


def _build_accepted_compound_anchor_index(
    clusters: list[MentionCluster],
    classifications: dict[str, ClassificationDecision],
) -> dict[int, list[tuple[int, int, str, int, str]]]:
    """Build per-span accepted compound index for lineage coverage checks."""
    index: dict[int, list[tuple[int, int, str, int, str]]] = defaultdict(list)
    for cluster in clusters:
        decision = classifications[cluster.normalized_key]
        if not decision.entityhood.accepted:
            continue
        part_count = len(compound_parts(cluster))
        for anchor in cluster.anchors:
            index[anchor.span_ordinal].append(
                (
                    anchor.start_char,
                    anchor.end_char,
                    cluster.cluster_id,
                    part_count,
                    cluster.normalized_key,
                )
            )
    return dict(index)


def _build_classification_trace(
    classification: ClassificationDecision,
) -> DocumentEntityClassificationTrace:
    """Map classification-stage output into the record-local trace type."""
    return DocumentEntityClassificationTrace(
        winning_score=classification.winning_score,
        runner_up_category=classification.runner_up_category,
        runner_up_score=classification.runner_up_score,
        evidence_by_category={
            category: CategoryEvidenceTrace(
                category=evidence.category,
                score=evidence.score,
                reasons=list(evidence.reasons),
                vetoes=list(evidence.vetoes),
            )
            for category, evidence in classification.evidence_by_category.items()
        },
        entityhood=EntityhoodTrace(
            score=classification.entityhood.score,
            accepted=classification.entityhood.accepted,
            reasons=list(classification.entityhood.reasons),
            weaknesses=list(classification.entityhood.weaknesses),
        ),
    )


def summarize_document_entities(
    pre: PreprocessedDocument,
    clusters: list[MentionCluster],
    bundle: PromotedEvidenceBundle,
    scores: dict[str, tuple[ConfidenceSignals, float]],
    classifications: dict[str, ClassificationDecision],
    attribution_records: list[AttributionRecord],
) -> list[DocumentEntityRecord]:
    """Build stable per-document entity summaries from pipeline outputs."""
    promoted_by_key = {
        candidate.cluster.normalized_key: candidate
        for candidate in bundle.promoted
    }
    review_by_key = {
        candidate.cluster.normalized_key: candidate
        for candidate in bundle.review_only
    }
    suppressed_by_key = {
        candidate.cluster.normalized_key: candidate
        for candidate in bundle.suppressed
    }

    windows_by_key: dict[str, list] = {}
    for window in bundle.evidence_windows:
        windows_by_key.setdefault(window.entity_key, []).append(window)

    quote_ranges_by_span: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for quote in pre.quote_spans:
        quote_ranges_by_span[quote.span_ordinal].append((quote.start_char, quote.end_char))

    sentence_by_span: dict[int, list] = defaultdict(list)
    sentence_initials_by_span: dict[int, set[int]] = defaultdict(set)
    for sentence in pre.sentences:
        sentence_by_span[sentence.span_ordinal].append(sentence)
        if sentence.tokens:
            sentence_initials_by_span[sentence.span_ordinal].add(sentence.tokens[0].start_char)

    attributed_quote_ranges_by_span: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for attribution in attribution_records:
        attributed_quote_ranges_by_span[attribution.quote_anchor.span_ordinal].append(
            (attribution.quote_anchor.start_char, attribution.quote_anchor.end_char)
        )

    accepted_compound_anchors_by_span = _build_accepted_compound_anchor_index(clusters, classifications)
    all_keys = {cluster.normalized_key for cluster in clusters}

    records: list[DocumentEntityRecord] = []
    for cluster in clusters:
        key = cluster.normalized_key
        signals, score = scores[key]
        classification = classifications[key]

        if key in promoted_by_key:
            bucket = DocumentEntityBucket.PROMOTED
            bucket_detail = ""
        elif key in review_by_key:
            bucket = DocumentEntityBucket.REVIEW_ONLY
            bucket_detail = review_by_key[key].reason
        else:
            bucket = DocumentEntityBucket.SUPPRESSED
            bucket_detail = suppressed_by_key[key].detail

        in_quote_count = 0
        non_quote_count = 0
        sentence_initial_count = 0
        address_like_count = 0
        attributed_speaker_nearby_count = 0
        one_token_utterance_count = 0
        candidate_parent_keys: set[str] = set()
        covered_anchor_count = 0

        part_count = len(compound_parts(cluster))
        appears_as_compound_surface = part_count > 1
        appears_as_compound_component = any(
            key in other_key.split() and other_key != key
            for other_key in all_keys
        )

        for anchor in cluster.anchors:
            ranges = quote_ranges_by_span.get(anchor.span_ordinal, [])
            enclosing_quotes = [
                (start, end)
                for start, end in ranges
                if start <= anchor.start_char and anchor.end_char <= end
            ]
            if enclosing_quotes:
                in_quote_count += 1
            else:
                non_quote_count += 1

            if anchor.start_char in sentence_initials_by_span.get(anchor.span_ordinal, set()):
                sentence_initial_count += 1

            quote_tokens = []
            for sentence in sentence_by_span.get(anchor.span_ordinal, []):
                if sentence.start_char <= anchor.start_char and anchor.end_char <= sentence.end_char:
                    try:
                        token_index = next(
                            index
                            for index, token in enumerate(sentence.tokens)
                            if token.start_char == anchor.start_char
                        )
                    except StopIteration:
                        continue
                    if is_address_like_reference(sentence, token_index, ranges):
                        address_like_count += 1

                    for quote_start, quote_end in enclosing_quotes:
                        quote_tokens = [
                            token for token in sentence.tokens
                            if quote_start <= token.start_char and token.end_char <= quote_end and any(ch.isalpha() for ch in token.text)
                        ]
                        if len(quote_tokens) == 1:
                            token = quote_tokens[0]
                            if token.start_char == anchor.start_char and token.end_char == anchor.end_char:
                                one_token_utterance_count += 1
                    break

            attributed_ranges = attributed_quote_ranges_by_span.get(anchor.span_ordinal, [])
            if any(start <= anchor.start_char and anchor.end_char <= end for start, end in attributed_ranges):
                attributed_speaker_nearby_count += 1

            candidates = accepted_compound_anchors_by_span.get(anchor.span_ordinal, [])
            covering = [
                (other_key, other_part_count)
                for other_start, other_end, other_cluster_id, other_part_count, other_key in candidates
                if other_cluster_id != cluster.cluster_id
                and other_part_count > part_count
                and other_start <= anchor.start_char
                and other_end >= anchor.end_char
            ]
            if covering:
                covered_anchor_count += 1
                for other_key, _ in covering:
                    candidate_parent_keys.add(other_key)

        uncovered_anchor_count = len(cluster.anchors) - covered_anchor_count

        records.append(DocumentEntityRecord(
            identity=DocumentEntityIdentity(
                record_id=stable_hash_id(
                    "document_entity_record",
                    key,
                    pre.source.path,
                ),
                document_anchor=DocumentAnchor(path=pre.source.path),
                normalized_key=key,
                surface_forms=sorted(cluster.surface_forms),
            ),
            current_state=DocumentEntityCurrentState(
                winning_category=classification.winning_category,
                resolved=classification.resolved,
                bucket=bucket,
            ),
            source_evidence=DocumentEntitySourceEvidence(
                occurrence_count=cluster.occurrence_count,
                anchors=list(cluster.anchors),
                evidence_windows=windows_by_key.get(key, []),
            ),
            classification_trace=_build_classification_trace(classification),
            promotion_trace=DocumentEntityPromotionTrace(
                confidence_score=score,
                suppression_reason=suppressed_by_key[key].reason if bucket == DocumentEntityBucket.SUPPRESSED else None,
                bucket_detail=bucket_detail,
                rule_tier=signals.rule_tier,
                scene_count=signals.scene_count,
                attribution_count=signals.attribution_count,
                possessive_count=signals.possessive_count,
                tfidf_score=signals.tfidf_score,
            ),
            discourse_profile=DocumentEntityDiscourseProfile(
                in_quote_count=in_quote_count,
                non_quote_count=non_quote_count,
                quote_only=(in_quote_count > 0 and non_quote_count == 0),
                sentence_initial_count=sentence_initial_count,
                sentence_initial_only=(sentence_initial_count > 0 and sentence_initial_count == len(cluster.anchors)),
                address_like_count=address_like_count,
                attributed_speaker_nearby_count=attributed_speaker_nearby_count,
                one_token_utterance_count=one_token_utterance_count,
            ),
            support_profile=DocumentEntitySupportProfile(
                title_support_count=cluster.title_support_count,
                possessive_support_count=cluster.possessive_support_count,
                location_support_count=cluster.location_support_count,
                linked_field_count=len(cluster.linked_fields),
                linked_definition_count=len(cluster.linked_definitions),
                linked_seed_count=len(cluster.linked_seeds),
            ),
            lineage_profile=DocumentEntityLineageProfile(
                compound_part_count=part_count,
                fully_covered_by_longer_compound=(bool(cluster.anchors) and uncovered_anchor_count == 0),
                candidate_parent_keys=sorted(candidate_parent_keys),
                covered_anchor_count=covered_anchor_count,
                uncovered_anchor_count=uncovered_anchor_count,
                appears_as_compound_component=appears_as_compound_component,
                appears_as_compound_surface=appears_as_compound_surface,
            ),
        ))

    _attach_suppressed_evidence_to_records(records)

    return records
