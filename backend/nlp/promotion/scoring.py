"""
Confidence scoring for mention clusters.

Computes per-cluster TF-IDF specificity and combines it with structural signals
(rule tier, title presence, possessive count, attribution count, scene dispersion)
into a single deterministic confidence score in [0.0, 1.0].

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster list] --> B[compute_tfidf\nper cluster]
        C[BootstrappedLexiconEntry list] --> D[Build lexicon key set]
        E[AttributionRecord list] --> F[Count attributions per cluster key]
        B & D & F --> G[For each cluster: compute_signals]
        G --> H[score_cluster: weighted sum of signals]
        H --> I[dict: normalized_key -> ConfidenceSignals + float]
"""

from __future__ import annotations

import math

from backend.nlp.types import (
    BootstrappedLexiconEntry,
    ConfidenceSignals,
    MentionCluster,
    PreprocessedDocument,
)
from backend.nlp.promotion.attribution import AttributionRecord

# ---------------------------------------------------------------------------
# Promotion thresholds - exported so promotion.py and tests can reference them
# without duplicating the values.
# ---------------------------------------------------------------------------

# Clusters with score >= PROMOTE_THRESHOLD are emitted as PromotedCandidate.
PROMOTE_THRESHOLD: float = 0.45

# Clusters with score < SUPPRESS_THRESHOLD are emitted as SuppressedCandidate
# with reason LOW_CONFIDENCE.
SUPPRESS_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Scoring weights
#
# Rule tier is the dominant signal. Supporting signals have diminishing-returns
# caps so that a cluster cannot reach promotion through supporting signals alone
# without a meaningful rule tier.
# ---------------------------------------------------------------------------

# Base score indexed by rule tier (0 = invalid, 1-3 = valid tiers).
_TIER_BASE: list[float] = [0.0, 0.15, 0.30, 0.50]

_TITLE_BONUS: float = 0.10        # title prefix present in any mention
_POSSESSIVE_WEIGHT: float = 0.05  # per distinct possessive surface form
_POSSESSIVE_CAP: float = 0.10
_ATTRIBUTION_WEIGHT: float = 0.10 # per speech-verb attribution record
_ATTRIBUTION_CAP: float = 0.20
_SCENE_WEIGHT: float = 0.05       # per distinct scene with at least one mention
_SCENE_CAP: float = 0.15
_TFIDF_WEIGHT: float = 0.10       # scaled TF-IDF specificity score
_TFIDF_CAP: float = 0.10


def compute_tfidf(
    clusters: list[MentionCluster],
    pre: PreprocessedDocument,
) -> dict[str, float]:
    """Compute TF-IDF specificity score for each cluster.

    Term frequency is the cluster's occurrence count relative to the total
    across all clusters in the document. Inverse document frequency uses a
    cascade of structural units to find the finest granularity available:

    1. Scenes (--- bounded): preferred for most fiction manuscripts.
    2. Sections (# bounded): fallback for structured documents that use
       chapter headings but no explicit scene breaks.
    3. Content spans (paragraphs + headings): final fallback when a document
       has no structural markers at all. Span ordinals are read directly from
       each anchor so no character-range search is needed.

    A single unit of any type produces zero IDF for all clusters regardless
    of formula, so the cascade falls through to finer granularity until a
    level with more than one unit is found.

    Args:
        clusters: All clusters from the current bootstrap pass.
        pre: The preprocessed document, used to access structural boundaries.

    Returns:
        Mapping from normalized_key to TF-IDF score, clamped to >= 0.0.
    """
    scenes = pre.source.scenes
    sections = pre.source.sections

    # unit_triples: list of (start_char, end_char, unit_id) for character-range
    # lookup. Set to None when the span fallback is used instead, because span
    # ordinals are available directly on anchors without a range search.
    if len(scenes) > 1:
        unit_triples = [(s.start_char, s.end_char, s.scene_index) for s in scenes]
        num_units = len(scenes)
    elif len(sections) > 1:
        unit_triples = [(s.start_char, s.end_char, s.section_index) for s in sections]
        num_units = len(sections)
    else:
        unit_triples = None
        num_units = max(len(pre.tokens_by_span), 1)

    total_count = max(sum(c.occurrence_count for c in clusters), 1)

    result: dict[str, float] = {}
    for cluster in clusters:
        tf = cluster.occurrence_count / total_count

        if unit_triples is not None:
            units_with_cluster: set[int] = set()
            for anchor in cluster.anchors:
                for u_start, u_end, u_id in unit_triples:
                    if u_start <= anchor.start_char < u_end:
                        units_with_cluster.add(u_id)
                        break
            df = len(units_with_cluster)
        else:
            # Each anchor carries its parent span_ordinal, so df is simply the
            # count of distinct content spans this cluster appears in.
            df = len({anchor.span_ordinal for anchor in cluster.anchors})

        idf = math.log(num_units / (1 + df))
        result[cluster.normalized_key] = max(0.0, tf * idf)

    return result


def compute_signals(
    cluster: MentionCluster,
    lexicon_keys: set[str],
    attr_counts: dict[str, int],
    pre: PreprocessedDocument,
    tfidf_scores: dict[str, float],
) -> ConfidenceSignals:
    """Compute the ConfidenceSignals for a single cluster.

    Args:
        cluster: The cluster to score.
        lexicon_keys: Set of normalized_phrase values present in the final lexicon.
            Membership here gives rule_tier=3 (strongest evidence).
        attr_counts: Mapping from normalized_key to number of attribution records.
        pre: The preprocessed document, used for scene-boundary lookups.
        tfidf_scores: Pre-computed TF-IDF scores from compute_tfidf.

    Returns:
        ConfidenceSignals with all deterministic signal values populated.
    """
    # Lexicon membership outranks title, which outranks bare capitalisation.
    # A cluster in the lexicon was inducted through strong recurrence or
    # structural evidence and is the most reliable signal we have.
    if cluster.normalized_key in lexicon_keys:
        rule_tier = 3
    elif cluster.has_title_support:
        rule_tier = 2
    else:
        rule_tier = 1

    # Count distinct surface forms that end with the possessive suffix rather
    # than using the boolean flag, to get a quantitative signal.
    possessive_count = sum(
        1 for sf in cluster.surface_forms
        if sf.endswith("'s") or sf.endswith("s'")
    )

    # Count distinct scenes that contain at least one mention anchor.
    scenes = pre.source.scenes
    scene_indices: set[int] = set()
    for anchor in cluster.anchors:
        for scene in scenes:
            if scene.start_char <= anchor.start_char < scene.end_char:
                scene_indices.add(scene.scene_index)
                break
    # A cluster with anchors but in a document with no explicit scene breaks
    # (empty scenes list) is treated as appearing in one implicit scene.
    scene_count = max(len(scene_indices), 1 if cluster.anchors else 0)

    return ConfidenceSignals(
        rule_tier=rule_tier,
        has_title=cluster.has_title_support,
        possessive_count=possessive_count,
        attribution_count=attr_counts.get(cluster.normalized_key, 0),
        scene_count=scene_count,
        tfidf_score=tfidf_scores.get(cluster.normalized_key, 0.0),
    )


def score_cluster(signals: ConfidenceSignals) -> float:
    """Compute a deterministic confidence score from pre-computed signals.

    Rule tier is the dominant component. Supporting signals (title, possessives,
    attributions, scene dispersion, TF-IDF) add on top with diminishing-returns
    caps so that no single supporting signal can push a tier-1 cluster into the
    promotion band without at least two others.

    Args:
        signals: Signal values from compute_signals.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    score = _TIER_BASE[min(signals.rule_tier, 3)]

    if signals.has_title:
        score += _TITLE_BONUS

    score += min(signals.possessive_count * _POSSESSIVE_WEIGHT, _POSSESSIVE_CAP)
    score += min(signals.attribution_count * _ATTRIBUTION_WEIGHT, _ATTRIBUTION_CAP)
    score += min(signals.scene_count * _SCENE_WEIGHT, _SCENE_CAP)
    score += min(signals.tfidf_score * _TFIDF_WEIGHT, _TFIDF_CAP)

    return min(score, 1.0)


def score_all(
    clusters: list[MentionCluster],
    lexicon: list[BootstrappedLexiconEntry],
    attribution_records: list[AttributionRecord],
    pre: PreprocessedDocument,
) -> dict[str, tuple[ConfidenceSignals, float]]:
    """Score all clusters and return their signals and confidence scores.

    Args:
        clusters: All clusters from the bootstrap result.
        lexicon: Final lexicon entries. A cluster whose normalized_key appears
            here gets rule_tier=3.
        attribution_records: Speaker attribution records from attribute_dialogue.
        pre: The preprocessed document, used for TF-IDF and scene counting.

    Returns:
        Mapping from normalized_key to (ConfidenceSignals, confidence_score).
    """
    lexicon_keys = {e.normalized_phrase for e in lexicon}

    attr_counts: dict[str, int] = {}
    for record in attribution_records:
        attr_counts[record.speaker_key] = attr_counts.get(record.speaker_key, 0) + 1

    tfidf_scores = compute_tfidf(clusters, pre)

    result: dict[str, tuple[ConfidenceSignals, float]] = {}
    for cluster in clusters:
        signals = compute_signals(cluster, lexicon_keys, attr_counts, pre, tfidf_scores)
        score = score_cluster(signals)
        result[cluster.normalized_key] = (signals, score)

    return result
