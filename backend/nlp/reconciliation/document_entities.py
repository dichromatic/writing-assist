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

    return records
