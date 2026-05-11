"""
Per-document entity summaries for later corpus reconciliation.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster list] --> B[score_all]
        C[AttributionRecord list] --> B
        A --> D[classify_clusters]
        C --> D
        E[PromotedEvidenceBundle] --> F[Bucket lookup]
        B & D & F --> G[DocumentEntityRecord list]
"""

from __future__ import annotations

from backend.nlp.classification import classify_clusters
from backend.nlp.promotion.scoring import score_all
from backend.nlp.types import (
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    MentionCluster,
    PreprocessedDocument,
    PromotedEvidenceBundle,
    SpanAnchor,
    SuppressedEvidence,
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
    assert record.suppression_reason is not None
    return SuppressedEvidence(
        document_anchor=record.document_anchor,
        normalized_key=record.normalized_key,
        surface_forms=list(record.surface_forms),
        winning_category=record.winning_category,
        confidence_score=record.confidence_score,
        reason=record.suppression_reason,
        detail=record.bucket_detail,
        anchors=list(record.anchors),
    )


def _attach_suppressed_evidence_to_records(
    records: list[DocumentEntityRecord],
) -> None:
    """Attach suppressed records beneath stronger overlapping local entities.

    The later semantic pass should still be able to inspect weak fragments,
    title-like debris, and overlap noise, but not as a flat document-wide pile.
    This helper keeps the attachment rule intentionally simple and auditable:
    a suppressed record only attaches to a non-suppressed record when their
    anchors overlap in the same span, with full containment preferred over
    partial overlap.

    Args:
        records: Document-local entity records to enrich in place.
    """
    survivors = [
        record for record in records
        if record.bucket != DocumentEntityBucket.SUPPRESSED
    ]
    suppressed = [
        record for record in records
        if record.bucket == DocumentEntityBucket.SUPPRESSED
    ]

    for suppressed_record in suppressed:
        best_target: DocumentEntityRecord | None = None
        best_score: tuple[int, int, float, int, str] | None = None
        for target in survivors:
            containment_count = 0
            overlap_count = 0
            for suppressed_anchor in suppressed_record.anchors:
                for target_anchor in target.anchors:
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
                target.confidence_score,
                target.occurrence_count,
                target.normalized_key,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_target = target

        if best_target is None:
            continue

        best_target.suppressed_related_evidence.append(
            _build_suppressed_evidence(suppressed_record)
        )


def summarize_document_entities(
    pre: PreprocessedDocument,
    clusters: list[MentionCluster],
    attribution_records: list,
    bundle: PromotedEvidenceBundle,
) -> list[DocumentEntityRecord]:
    """Build stable per-document entity summaries from pipeline outputs.

    The document extractor already decided promotion bucket, class resolution,
    and confidence for each cluster. This function preserves those local
    decisions in a uniform record format so later corpus stages can compare
    documents without re-deriving bucket state ad hoc.

    Args:
        pre: Preprocessed document context.
        clusters: Final clusters for the document.
        attribution_records: Dialogue attribution records.
        bundle: Final promotion bundle for the same document.

    Returns:
        DocumentEntityRecord list in cluster order.
    """
    scores = score_all(clusters, attribution_records, pre)
    classifications = classify_clusters(clusters, pre, attribution_records)

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

        records.append(DocumentEntityRecord(
            document_anchor=DocumentAnchor(path=pre.source.path),
            normalized_key=key,
            surface_forms=sorted(cluster.surface_forms),
            winning_category=classification.winning_category,
            resolved=classification.resolved,
            entityhood_score=classification.entityhood.score,
            entityhood_accepted=classification.entityhood.accepted,
            confidence_score=score,
            bucket=bucket,
            suppression_reason=suppressed_by_key[key].reason if bucket == DocumentEntityBucket.SUPPRESSED else None,
            bucket_detail=bucket_detail,
            occurrence_count=cluster.occurrence_count,
            rule_tier=signals.rule_tier,
            scene_count=signals.scene_count,
            attribution_count=signals.attribution_count,
            has_title_support=cluster.has_title_support,
            has_possessive_support=cluster.has_possessive_support,
            anchors=list(cluster.anchors),
            evidence_windows=windows_by_key.get(key, []),
        ))

    _attach_suppressed_evidence_to_records(records)

    return records
