"""
Evidence promotion - classifies clusters into promoted, review-only, or suppressed
buckets and constructs the PromotedEvidenceBundle for downstream stages.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument + Clusters + Lexicon + AttributionRecords] --> B[score_all]
        B --> C{For each cluster}
        C -->|is stopword| D[SuppressedCandidate STOPWORD]
        C -->|score below SUPPRESS_THRESHOLD| E[SuppressedCandidate LOW_CONFIDENCE]
        C -->|score >= PROMOTE_THRESHOLD\nAND not bare title key| F[PromotedCandidate]
        C -->|score >= PROMOTE_THRESHOLD\nAND is bare title key| G[ReviewOnlyCandidate\nreason: bare title prefix]
        C -->|score between thresholds| G2[ReviewOnlyCandidate\nreason: mid-confidence]
        F & G & G2 --> H[Build EvidenceWindows for all anchors]
        D & E & F & G & G2 & H --> I[PromotedEvidenceBundle]
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.nlp.classification.arbitration import classify_clusters
from backend.nlp.classification.compound_shapes import compound_parts
from backend.nlp.types import (
    BootstrappedLexiconEntry,
    DocumentAnchor,
    EvidenceWindow,
    MentionCluster,
    LexiconCategory,
    PreprocessedDocument,
    PromotedCandidate,
    PromotedEvidenceBundle,
    ReviewOnlyCandidate,
    Sentence,
    SpanAnchor,
    SuppressReason,
    SuppressedCandidate,
)
from backend.nlp.harvesting.shared import (
    TITLE_PREFIXES_LOWER,
    has_generic_modifier_profile,
    has_generic_verb_sense,
    is_stopword,
)
from backend.nlp.promotion.attribution import AttributionRecord
from backend.nlp.promotion.scoring import PROMOTE_THRESHOLD, SUPPRESS_THRESHOLD, score_all

# Maximum characters of surrounding context to include in each EvidenceWindow.
# The actual window is snapped inward to the nearest sentence boundary, so
# context strings always contain complete sentences. 150 characters is the
# outer bound; the real edge is the sentence start or end closest to that limit.
_CONTEXT_RADIUS = 150

@dataclass
class PromotionResult:
    """Complete promotion output including reusable intermediate decisions.

    Args:
        bundle: Final promotion buckets and evidence windows.
        classifications: Deterministic category decisions per cluster key.
        scores: Deterministic confidence signals and score per cluster key.
    """

    bundle: PromotedEvidenceBundle
    classifications: dict[str, object]
    scores: dict[str, tuple[object, float]]


def _should_suppress_generic_verb_noise(
    cluster: MentionCluster,
    classification,
    signals,
) -> bool:
    """Return True when a weak single-token cluster is ordinary verb noise.

    This guard runs late in promotion rather than at harvest time so real
    entities still have a chance to accumulate title, possessive, attribution,
    or reference-note support first. Only unresolved bare single tokens with
    no structural backing are eligible.
    """
    return (
        " " not in cluster.normalized_key
        and classification.winning_category == LexiconCategory.UNRESOLVED
        and signals.rule_tier <= 2
        and signals.attribution_count == 0
        and not cluster.has_title_support
        and not cluster.linked_fields
        and not cluster.linked_definitions
        and not cluster.linked_seeds
        and has_generic_verb_sense(cluster.normalized_key)
    )


def _has_fully_covering_longer_compound(
    cluster: MentionCluster,
    accepted_compound_anchors_by_span: dict[int, list[tuple[int, int, str, int]]],
) -> bool:
    """Return True when every anchor is covered by a longer accepted compound.

    This is a structural overlap check used to suppress weak component-only
    survivors such as ``old`` beneath ``old man hiroshi``. A longer compound
    only counts when it has at least accepted entityhood so arbitrary overlaps
    do not suppress shorter clusters.
    """
    cluster_part_count = len(compound_parts(cluster))
    for anchor in cluster.anchors:
        candidates = accepted_compound_anchors_by_span.get(anchor.span_ordinal, [])
        if not any(
            other_cluster_id != cluster.cluster_id
            and other_part_count > cluster_part_count
            and other_start <= anchor.start_char
            and other_end >= anchor.end_char
            for other_start, other_end, other_cluster_id, other_part_count in candidates
        ):
            return False
    return bool(cluster.anchors)


def _should_suppress_component_overlap_noise(
    cluster: MentionCluster,
    accepted_compound_anchors_by_span: dict[int, list[tuple[int, int, str, int]]],
    classifications: dict[str, object],
    signals,
) -> bool:
    """Return True for generic component-only clusters covered by compounds.

    This is intentionally narrow. The shorter cluster must be generic-looking,
    must have no independent structural backing, and must be fully explained by
    overlap with a longer accepted compound. Specific name components such as
    ``tsushima`` and ``yoshiko`` stay available for later alias reconciliation.
    """
    parts = compound_parts(cluster)
    if not parts:
        return False

    if cluster.has_title_support or cluster.linked_fields or cluster.linked_definitions or cluster.linked_seeds:
        return False
    if signals.attribution_count > 0:
        return False
    if not _has_fully_covering_longer_compound(cluster, accepted_compound_anchors_by_span):
        return False

    decision = classifications[cluster.normalized_key]
    if decision.winning_score > 0.65:
        return False

    return has_generic_modifier_profile(parts[0])


def _build_accepted_compound_anchor_index(
    clusters: list[MentionCluster],
    classifications: dict[str, object],
) -> dict[int, list[tuple[int, int, str, int]]]:
    """Build per-span accepted compound anchor index for overlap checks.

    Args:
        clusters: Mention clusters under promotion.
        classifications: Deterministic classification decisions by cluster key.

    Returns:
        Mapping:
        - key: span ordinal
        - value: list of tuples
          (anchor_start, anchor_end, cluster_id, compound_part_count)
    """
    index: dict[int, list[tuple[int, int, str, int]]] = {}
    for cluster in clusters:
        decision = classifications[cluster.normalized_key]
        if not decision.entityhood.accepted:
            continue
        part_count = len(compound_parts(cluster))
        for anchor in cluster.anchors:
            index.setdefault(anchor.span_ordinal, []).append(
                (
                    anchor.start_char,
                    anchor.end_char,
                    cluster.cluster_id,
                    part_count,
                )
            )
    return index


def _should_promote_place(
    cluster: MentionCluster,
    classification,
    signals,
    score: float,
) -> bool:
    """Return True when a resolved place has enough support to auto-promote."""
    return (
        classification.winning_category == LexiconCategory.PLACE
        and classification.resolved
        and classification.entityhood.accepted
        and classification.winning_score >= 0.60
        and cluster.occurrence_count >= 5
        and signals.scene_count >= 2
        and score >= 0.45
    )


def _review_reason_for_classification(
    cluster: MentionCluster,
    classification,
    signals,
    score: float,
) -> str | None:
    """Return a class-aware review reason when promotion should be withheld."""
    category = classification.winning_category

    if category == LexiconCategory.PLACE:
        if _should_promote_place(cluster, classification, signals, score):
            return None
        return (
            f"resolved as {category.value}; confidence {score:.3f} requires "
            f"place review until broader place support is present"
        )

    if category == LexiconCategory.UNRESOLVED:
        return (
            f"'{category.value}' classification remained unresolved; "
            f"confidence {score:.3f} requires human review"
        )

    if category in {
        LexiconCategory.GROUP,
        LexiconCategory.OBJECT,
        LexiconCategory.EVENT,
        LexiconCategory.CONCEPT,
    }:
        return (
            f"resolved as {category.value}; confidence {score:.3f} requires "
            f"class-aware review before promotion"
        )

    return None


def _build_evidence_windows(
    cluster: MentionCluster,
    attribution_records: list[AttributionRecord],
    raw_text: str,
    sentences: list[Sentence],
) -> list[EvidenceWindow]:
    """Build an EvidenceWindow for each anchor in the cluster.

    Anchors are sorted by document position. The first anchor (earliest
    start_char) is marked as the first introduction. For each anchor, the
    span_ordinal is looked up in the attribution map to find who (if anyone)
    was speaking in that span. The speaker may be this entity, another entity,
    or nobody. has_attribution reflects the presence of any attribution in the
    span, not only cases where this entity itself spoke.

    Context strings are snapped to sentence boundaries within _CONTEXT_RADIUS
    so that context_before always starts at a sentence boundary and
    context_after always ends at a sentence boundary. This produces cleaner
    context for retrieval than a raw character slice, which often begins or
    ends mid-sentence. If no sentence boundary falls within the radius, the
    raw character limit is used as a fallback.

    Args:
        cluster: The cluster whose anchors become evidence windows.
        attribution_records: All attribution records from the attribution stage.
        raw_text: The full raw document text used to extract context strings.
        sentences: All sentences from the preprocessed document, used to snap
            context boundaries to sentence edges.

    Returns:
        EvidenceWindow records in document order (ascending start_char).
    """
    if not cluster.anchors:
        return []

    sorted_anchors = sorted(cluster.anchors, key=lambda a: a.start_char)
    first_start = sorted_anchors[0].start_char

    # Map each span_ordinal to the speaker key of the attribution recorded there.
    # When multiple attributions fall in the same span, the case where this
    # entity itself is the speaker takes priority (most informative for retrieval).
    # Otherwise the first attribution in the list wins.
    span_to_speaker: dict[int, str] = {}
    for r in attribution_records:
        ordinal = r.quote_anchor.span_ordinal
        if ordinal not in span_to_speaker or r.speaker_key == cluster.normalized_key:
            span_to_speaker[ordinal] = r.speaker_key

    windows: list[EvidenceWindow] = []
    for anchor in sorted_anchors:
        # Compute the raw character boundaries first, then snap each edge
        # inward to the nearest sentence boundary within the radius.
        before_raw = max(0, anchor.start_char - _CONTEXT_RADIUS)
        after_raw = min(len(raw_text), anchor.end_char + _CONTEXT_RADIUS)

        # Snap the before edge to the start of the first sentence that begins
        # within the radius window. This avoids starting context mid-sentence.
        before_start = before_raw
        for sentence in sentences:
            if before_raw <= sentence.start_char < anchor.start_char:
                before_start = sentence.start_char
                break

        # Snap the after edge to the end of the last sentence that ends within
        # the radius window. This avoids ending context mid-sentence.
        after_end = after_raw
        for sentence in reversed(sentences):
            if anchor.end_char < sentence.end_char <= after_raw:
                after_end = sentence.end_char
                break

        context_before = raw_text[before_start:anchor.start_char]
        context_after = raw_text[anchor.end_char:after_end]
        # speaker is whoever was attributed as speaker in this span - which may be
        # this entity, another entity, or nobody. has_attribution is True whenever
        # any attribution was detected in the span, not only when this entity spoke.
        speaker_in_span = span_to_speaker.get(anchor.span_ordinal)

        windows.append(EvidenceWindow(
            entity_key=cluster.normalized_key,
            anchor=anchor,
            context_before=context_before,
            context_after=context_after,
            is_first_introduction=(anchor.start_char == first_start),
            has_attribution=speaker_in_span is not None,
            speaker=speaker_in_span,
        ))

    return windows


def promote(
    pre: PreprocessedDocument,
    clusters: list[MentionCluster],
    lexicon: list[BootstrappedLexiconEntry],
    attribution_records: list[AttributionRecord],
) -> PromotionResult:
    """Classify clusters and return the complete PromotedEvidenceBundle.

    Applies structural suppression (stopword check) before score-based
    classification. Score thresholds are imported from scoring.py so the
    boundary values are defined in one place.

    Args:
        pre: The preprocessed document, needed by score_all for TF-IDF and
            scene boundary lookups.
        clusters: All clusters from the bootstrap result's final pass.
        lexicon: The final bootstrapped lexicon from the convergence loop.
        attribution_records: Speaker attribution records from attribute_dialogue.

    Returns:
        PromotionResult with the promotion bundle plus reusable score and
        classification maps for downstream stages.
    """
    doc = pre.source
    scores = score_all(clusters, attribution_records, pre)
    classifications = classify_clusters(clusters, pre, attribution_records)
    accepted_compound_anchors_by_span = _build_accepted_compound_anchor_index(
        clusters,
        classifications,
    )

    promoted: list[PromotedCandidate] = []
    review_only: list[ReviewOnlyCandidate] = []
    suppressed: list[SuppressedCandidate] = []
    evidence_windows: list[EvidenceWindow] = []

    for cluster in clusters:
        # Stopword suppression runs before scoring as a structural safeguard.
        # Stopwords should have been filtered at harvest time, but this catch
        # prevents any that escaped from polluting the promoted or review buckets.
        if is_stopword(cluster.normalized_key):
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.STOPWORD,
                detail=f"'{cluster.normalized_key}' matched the stopword list",
            ))
            continue

        signals, score = scores[cluster.normalized_key]
        classification = classifications[cluster.normalized_key]

        if _should_suppress_component_overlap_noise(
            cluster,
            accepted_compound_anchors_by_span,
            classifications,
            signals,
        ):
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
                detail=(
                    f"'{cluster.normalized_key}' is fully covered by a longer "
                    f"compound and has no independent entity support"
                ),
            ))
            continue

        if _should_suppress_generic_verb_noise(cluster, classification, signals):
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.GENERIC_LEXICAL_NOISE,
                detail=(
                    f"'{cluster.normalized_key}' behaves like an ordinary verb "
                    f"lemma without entity support"
                ),
            ))
            continue

        # Weak unresolved clusters are usually capitalized noise that survived
        # harvesting through recurrence or scene dispersion alone. Keep
        # unresolved for plausible entities, but suppress clusters whose
        # entityhood never cleared the acceptance threshold.
        if (
            classification.winning_category == LexiconCategory.UNRESOLVED
            and not classification.entityhood.accepted
        ):
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.LOW_ENTITYHOOD,
                detail=(
                    f"entityhood {classification.entityhood.score:.3f} below "
                    f"acceptance threshold for unresolved cluster"
                ),
            ))
            continue

        review_reason = _review_reason_for_classification(
            cluster,
            classification,
            signals,
            score,
        )

        if score < SUPPRESS_THRESHOLD and review_reason is None:
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.LOW_CONFIDENCE,
                detail=(
                    f"confidence {score:.3f} below suppression threshold "
                    f"{SUPPRESS_THRESHOLD:.3f}"
                ),
            ))
            continue

        if (
            score >= PROMOTE_THRESHOLD
            and cluster.normalized_key not in TITLE_PREFIXES_LOWER
            and review_reason is None
        ):
            promoted.append(PromotedCandidate(
                cluster=cluster,
                confidence_score=score,
                signals=signals,
                anchor=DocumentAnchor(path=doc.path),
            ))
            evidence_windows.extend(
                _build_evidence_windows(cluster, attribution_records, doc.raw_text, pre.sentences)
            )
        else:
            # Mid-confidence clusters land here, and so do bare-title clusters
            # regardless of score. A bare title like "captain" or "lord" scores
            # well when it recurs frequently, but without an accompanying name it
            # is ambiguous - the same title could refer to different people in
            # different scenes. Human review is required before promoting.
            if score >= PROMOTE_THRESHOLD and cluster.normalized_key in TITLE_PREFIXES_LOWER:
                reason = (
                    f"'{cluster.normalized_key}' is a bare title prefix; "
                    f"confidence {score:.3f} is above promotion threshold "
                    f"{PROMOTE_THRESHOLD:.3f} but human review is required"
                )
            elif score >= PROMOTE_THRESHOLD and review_reason is not None:
                reason = review_reason
            else:
                reason = (
                    f"confidence {score:.3f} between suppression threshold "
                    f"{SUPPRESS_THRESHOLD:.3f} and promotion threshold "
                    f"{PROMOTE_THRESHOLD:.3f}"
                )
            review_only.append(ReviewOnlyCandidate(
                cluster=cluster,
                confidence_score=score,
                reason=reason,
            ))
            # Review-only clusters receive all windows, same as promoted.
            # A reviewer deciding whether a mid-confidence cluster is real needs
            # to see the spread of occurrences, not just the first one.
            evidence_windows.extend(
                _build_evidence_windows(cluster, attribution_records, doc.raw_text, pre.sentences)
            )

    return PromotionResult(
        bundle=PromotedEvidenceBundle(
            document_anchor=DocumentAnchor(path=doc.path),
            promoted=promoted,
            review_only=review_only,
            suppressed=suppressed,
            evidence_windows=evidence_windows,
        ),
        classifications=classifications,
        scores=scores,
    )
