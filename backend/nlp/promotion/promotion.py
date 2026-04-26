"""
Evidence promotion - classifies clusters into promoted, review-only, or suppressed
buckets and constructs the PromotedEvidenceBundle for downstream stages.

.. code-block:: mermaid

    flowchart TD
        A[PreprocessedDocument + Clusters + Lexicon + AttributionRecords] --> B[score_all]
        B --> C{For each cluster}
        C -->|is stopword| D[SuppressedCandidate STOPWORD]
        C -->|score below SUPPRESS_THRESHOLD| E[SuppressedCandidate LOW_CONFIDENCE]
        C -->|score >= PROMOTE_THRESHOLD\nAND is bare title key| G[ReviewOnlyCandidate BARE_TITLE]
        C -->|score >= PROMOTE_THRESHOLD\nAND not bare title key| F[PromotedCandidate]
        C -->|otherwise| G
        F & G --> H[Build EvidenceWindows for all anchors]
        D & E & F & G & H --> I[PromotedEvidenceBundle]
"""

from __future__ import annotations

from backend.nlp.types import (
    BootstrappedLexiconEntry,
    DocumentAnchor,
    EvidenceWindow,
    MentionCluster,
    PreprocessedDocument,
    PromotedCandidate,
    PromotedEvidenceBundle,
    ReviewOnlyCandidate,
    Sentence,
    SpanAnchor,
    SuppressReason,
    SuppressedCandidate,
)
from backend.nlp.harvesting.shared import is_stopword, TITLE_PREFIXES
from backend.nlp.promotion.attribution import AttributionRecord
from backend.nlp.promotion.scoring import PROMOTE_THRESHOLD, SUPPRESS_THRESHOLD, score_all

# Maximum characters of surrounding context to include in each EvidenceWindow.
# The actual window is snapped inward to the nearest sentence boundary, so
# context strings always contain complete sentences. 150 characters is the
# outer bound; the real edge is the sentence start or end closest to that limit.
_CONTEXT_RADIUS = 150

# Lowercased set of every title prefix string from shared.py. A cluster whose
# normalized_key appears here is a bare title ("captain", "lord") rather than a
# name. These clusters are real character references - characters are frequently
# referred to only by title - but they are ambiguous enough that a human reviewer
# should confirm them before they enter the promoted bucket.
_BARE_TITLE_KEYS: frozenset[str] = frozenset(t.lower() for t in TITLE_PREFIXES)


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
) -> PromotedEvidenceBundle:
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
        PromotedEvidenceBundle with all three classification buckets and
        the evidence windows drawn from promoted and review-only clusters.
    """
    doc = pre.source
    scores = score_all(clusters, lexicon, attribution_records, pre)

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

        if score < SUPPRESS_THRESHOLD:
            suppressed.append(SuppressedCandidate(
                cluster=cluster,
                reason=SuppressReason.LOW_CONFIDENCE,
                detail=(
                    f"confidence {score:.3f} below suppression threshold "
                    f"{SUPPRESS_THRESHOLD:.3f}"
                ),
            ))
            continue

        if score >= PROMOTE_THRESHOLD and cluster.normalized_key not in _BARE_TITLE_KEYS:
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
            if score >= PROMOTE_THRESHOLD and cluster.normalized_key in _BARE_TITLE_KEYS:
                reason = (
                    f"'{cluster.normalized_key}' is a bare title prefix; "
                    f"confidence {score:.3f} is above promotion threshold "
                    f"{PROMOTE_THRESHOLD:.3f} but human review is required"
                )
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

    return PromotedEvidenceBundle(
        document_anchor=DocumentAnchor(path=doc.path),
        promoted=promoted,
        review_only=review_only,
        suppressed=suppressed,
        evidence_windows=evidence_windows,
    )
