"""
Tests for backend/nlp/promotion/scoring.py and
        backend/nlp/promotion/promotion.py.

Each test encodes a non-obvious decision or invariant. Comments explain which
decision is locked in and why its breakage would be silent.
"""

import pytest

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.types import (
    BootstrappedLexiconEntry,
    ConfidenceSignals,
    LexiconCategory,
    MentionCluster,
    SpanAnchor,
    SuppressReason,
    stable_hash_id,
)
from backend.nlp.promotion.attribution import attribute_dialogue, AttributionRecord
from backend.nlp.promotion.scoring import (
    PROMOTE_THRESHOLD,
    SUPPRESS_THRESHOLD,
    score_cluster,
)
from backend.nlp.promotion.promotion import promote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_promote(text: str, path: str = "doc.md"):
    """Run the full pipeline through to a PromotedEvidenceBundle."""
    doc = parse(path, text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    return promote(pre, result.clusters, result.lexicon, attribution_records)


# ---------------------------------------------------------------------------
# Scoring: signal contribution and determinism
# ---------------------------------------------------------------------------

class TestScoring:
    def test_scene_dispersion_increases_score(self):
        # A cluster appearing in 3 scenes must score higher than an otherwise
        # identical cluster appearing in 1 scene. Scene dispersion is a strong
        # signal that the entity is a significant recurring presence, not an
        # incidental mention.
        sig_dispersed = ConfidenceSignals(
            rule_tier=2, has_title=True, possessive_count=0,
            attribution_count=0, scene_count=3, tfidf_score=0.0,
        )
        sig_singleton = ConfidenceSignals(
            rule_tier=2, has_title=True, possessive_count=0,
            attribution_count=0, scene_count=1, tfidf_score=0.0,
        )
        assert score_cluster(sig_dispersed) > score_cluster(sig_singleton)

    def test_tfidf_contributes_positively_to_score(self):
        # A higher TF-IDF specificity score must produce a higher confidence
        # score. TF-IDF downweights names that appear at background frequency
        # across all sections; specific names should score higher.
        sig_high = ConfidenceSignals(
            rule_tier=1, has_title=False, possessive_count=0,
            attribution_count=0, scene_count=1, tfidf_score=1.0,
        )
        sig_low = ConfidenceSignals(
            rule_tier=1, has_title=False, possessive_count=0,
            attribution_count=0, scene_count=1, tfidf_score=0.0,
        )
        assert score_cluster(sig_high) > score_cluster(sig_low)

    def test_attribution_count_increases_score(self):
        # A cluster attributed as speaker in dialogue must score higher than an
        # equivalent cluster with no attributions. Speaker attribution is strong
        # evidence of personhood.
        sig_attr = ConfidenceSignals(
            rule_tier=2, has_title=True, possessive_count=0,
            attribution_count=2, scene_count=1, tfidf_score=0.0,
        )
        sig_none = ConfidenceSignals(
            rule_tier=2, has_title=True, possessive_count=0,
            attribution_count=0, scene_count=1, tfidf_score=0.0,
        )
        assert score_cluster(sig_attr) > score_cluster(sig_none)

    def test_score_is_deterministic(self):
        # Given identical inputs, score_cluster must return the same value on
        # every call. Non-determinism would make the promotion boundary unstable
        # across runs and break any caching or comparison of scores.
        signals = ConfidenceSignals(
            rule_tier=2, has_title=True, possessive_count=1,
            attribution_count=1, scene_count=2, tfidf_score=0.3,
        )
        assert score_cluster(signals) == score_cluster(signals)

    def test_mid_confidence_signals_fall_in_review_only_band(self):
        # A tier-1 cluster with one possessive mention and no other signals
        # must land in the review-only score band. Without this invariant, the
        # middle bucket would be unreachable and all weak clusters would collapse
        # into either promoted or suppressed, losing the human-review mechanism.
        signals = ConfidenceSignals(
            rule_tier=1, has_title=False, possessive_count=1,
            attribution_count=0, scene_count=1, tfidf_score=0.0,
        )
        score = score_cluster(signals)
        assert SUPPRESS_THRESHOLD <= score < PROMOTE_THRESHOLD


# ---------------------------------------------------------------------------
# Promotion: classification and bundle structure
# ---------------------------------------------------------------------------

class TestPromotion:
    def test_titled_cluster_is_promoted(self):
        # A cluster with a title prefix that appears multiple times must land
        # in the promoted bucket. Titled names are the strongest deterministic
        # evidence of a character; failing to promote them defeats the purpose
        # of the harvesting stage.
        bundle = run_promote("She met Captain Aldous. Captain Aldous nodded.")
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        assert "aldous" in promoted_keys

    def test_weak_cluster_is_suppressed_with_low_confidence(self):
        # A bare-capitalised name appearing once mid-sentence with no title,
        # possessive, or recurrence must be suppressed as LOW_CONFIDENCE. Without
        # this check, every incidental capitalised word would survive to promoted
        # or review-only.
        bundle = run_promote("She saw Sunlight through the window.")
        suppressed_reasons = {sc.reason for sc in bundle.suppressed}
        assert SuppressReason.LOW_CONFIDENCE in suppressed_reasons

    def test_suppressed_candidate_has_non_empty_detail(self):
        # Every SuppressedCandidate must carry a non-empty detail string so the
        # suppression reason is traceable without re-running the pipeline. An
        # empty detail would make the suppression reason opaque to callers.
        bundle = run_promote("She saw Sunlight through the window.")
        for sc in bundle.suppressed:
            assert isinstance(sc.reason, SuppressReason)
            assert len(sc.detail) > 0

    def test_evidence_window_carries_source_anchor(self):
        # Every EvidenceWindow must have a SpanAnchor so retrieval can locate
        # the exact document position. A window without an anchor cannot be used
        # for context extraction and would silently break the retrieval interface.
        bundle = run_promote("She met Captain Aldous. Captain Aldous smiled.")
        for window in bundle.evidence_windows:
            assert window.anchor is not None
            assert isinstance(window.anchor, SpanAnchor)

    def test_first_introduction_flag_on_earliest_anchor(self):
        # The EvidenceWindow for the first occurrence of an entity must have
        # is_first_introduction=True. This flag guides retrieval to the passage
        # where the entity was introduced, which typically carries the richest
        # surrounding context.
        bundle = run_promote("Captain Aldous arrived. Captain Aldous smiled.")
        aldous_windows = [w for w in bundle.evidence_windows if w.entity_key == "aldous"]
        assert aldous_windows, "expected evidence windows for 'aldous'"
        first_start = min(w.anchor.start_char for w in aldous_windows)
        first_windows = [w for w in aldous_windows if w.anchor.start_char == first_start]
        assert any(w.is_first_introduction for w in first_windows)

    def test_attribution_visible_in_evidence_window(self):
        # When a cluster is identified as speaker in dialogue, at least one of
        # its EvidenceWindows must have has_attribution=True. Without this, the
        # retrieval layer cannot distinguish attributed dialogue from narration,
        # losing a key signal for character voice retrieval.
        text = '"Go now," Aldous said. Aldous arrived.'
        doc = parse("doc.md", text)
        pre = preprocess(doc)
        result = bootstrap(doc)
        attribution_records = attribute_dialogue(pre, result.clusters)
        bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
        aldous_windows = [w for w in bundle.evidence_windows if w.entity_key == "aldous"]
        assert any(w.has_attribution for w in aldous_windows)

    def test_evidence_window_context_snaps_to_sentence_boundary(self):
        # context_before must begin at the nearest sentence boundary within
        # _CONTEXT_RADIUS, not at a raw character offset that falls mid-sentence.
        # The long preceding sentence is wider than the 150-char radius, so a
        # raw slice would begin inside it. The snap locates the shorter sentence
        # that follows and uses its start as the left edge instead.
        long_sentence = (
            "The soldier had marched for many long and difficult days across "
            "the vast and seemingly endless plains without any rest at all, "
            "growing weaker with each passing hour."
        )  # ~165 chars - wider than _CONTEXT_RADIUS = 150
        text = (
            f"{long_sentence} He stopped. "
            "Captain Aldous arrived. Captain Aldous nodded."
        )
        bundle = run_promote(text)
        aldous_windows = [w for w in bundle.evidence_windows if w.entity_key == "aldous"]
        assert aldous_windows, "expected evidence windows for aldous"
        # Raw slice begins inside the long sentence (mid-word).
        # Snapped slice begins at "He stopped." - the first complete sentence
        # whose start falls within the radius window.
        before = aldous_windows[0].context_before
        assert before.strip().startswith("He stopped"), (
            f"context_before does not appear sentence-snapped: {repr(before[:60])}"
        )

    def test_bare_title_cluster_routes_to_review(self):
        # A cluster whose normalized key is a bare title prefix ("captain",
        # "lord") must land in review_only even when its score is above
        # PROMOTE_THRESHOLD. The same title can refer to different characters in
        # different scenes, so automated promotion would be ambiguous. Human
        # review is required to confirm the referent before it is promoted.
        bundle = run_promote(
            "The Captain arrived. The Captain sat down. The Captain spoke."
        )
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        assert "captain" in review_keys
        assert "captain" not in promoted_keys
