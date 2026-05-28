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
    result = bootstrap(doc, pre=pre)
    attribution_records = attribute_dialogue(pre, result.clusters)
    return promote(pre, result.clusters, result.lexicon, attribution_records).bundle


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

    def test_resolved_place_cluster_routes_to_review_not_promoted(self):
        # A place can accumulate enough generic confidence to cross the
        # promotion threshold, but resolved places should not be auto-promoted
        # on the same policy as characters. They need a category-aware review
        # path until place-specific promotion rules are defined.
        text = (
            "In Sidhe, bells rang. Sidhe's harbor glowed.\n\n"
            "---\n\n"
            "They returned to Sidhe.\n\n"
            "---\n\n"
            "Sidhe slept."
        )
        bundle = run_promote(text)
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "sidhe" not in promoted_keys
        assert "sidhe" in review_keys

    def test_strong_recurring_place_is_promoted(self):
        # Once a place has strong classification evidence plus broad document
        # recurrence, it should not be trapped in review forever. This is the
        # first positive place-promotion rule.
        text = (
            "In Tairngire, the sea glowed. Tairngire's capital shimmered. "
            "The forests of Tairngire hummed at dusk.\n\n"
            "---\n\n"
            "They returned to Tairngire. Tairngire waited beneath the storm.\n\n"
            "---\n\n"
            "Tairngire opened before them again."
        )
        bundle = run_promote(text)
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        assert "tairngire" in promoted_keys

    def test_thin_place_stays_review_only(self):
        # A place with strong local syntax but only a thin document footprint
        # should still route to review. The positive place path is meant for
        # recurring setting entities, not every one-off location mention.
        text = "In Sidhe, bells rang."
        bundle = run_promote(text)
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "sidhe" not in promoted_keys
        assert "sidhe" in review_keys

    def test_resolved_event_with_low_generic_score_routes_to_review(self):
        # A capitalized event can have enough typed evidence to resolve as
        # EVENT even when the generic promotion score is still below the usual
        # review band. It must remain visible for review rather than being
        # suppressed as low confidence.
        text = "The Festival began at dusk. During the Festival, bells rang."
        bundle = run_promote(text)
        suppressed_keys = {c.cluster.normalized_key for c in bundle.suppressed}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "festival" not in suppressed_keys
        assert "festival" in review_keys

    def test_resolved_concept_with_low_generic_score_routes_to_review(self):
        # Definition-style concept evidence should keep a concept visible for
        # review even when the generic promotion score remains low.
        text = (
            "The term Leva refers to a magical resonance system. "
            "Leva destabilized the chamber."
        )
        bundle = run_promote(text)
        suppressed_keys = {c.cluster.normalized_key for c in bundle.suppressed}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "leva" not in suppressed_keys
        assert "leva" in review_keys

    def test_resolved_group_with_low_generic_score_routes_to_review(self):
        # Group evidence should keep a resolved collective visible for review
        # even when the generic promotion score remains below the usual band.
        text = "She served with Meridian. Meridian deployed scouts at dawn."
        bundle = run_promote(text)
        suppressed_keys = {c.cluster.normalized_key for c in bundle.suppressed}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "meridian" not in suppressed_keys
        assert "meridian" in review_keys

    def test_accepted_unresolved_cluster_routes_to_review_not_promoted(self):
        # Plausible unresolved entities should survive for review, but they
        # should not auto-promote just because possessive recurrence and scene
        # dispersion pushed the generic score high enough.
        text = (
            "Aldous's sword fell.\n\n"
            "---\n\n"
            "Aldous's bag opened.\n\n"
            "---\n\n"
            "Aldous's room was dark."
        )
        bundle = run_promote(text)
        promoted_keys = {c.cluster.normalized_key for c in bundle.promoted}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "aldous" not in promoted_keys
        assert "aldous" in review_keys

    def test_weak_cluster_is_suppressed_with_low_confidence(self):
        # A bare-capitalised singleton with no structural support must be
        # suppressed before review. The entityhood guard is now the first
        # filter for this case, so callers can distinguish weak unresolved
        # noise from plausible mid-confidence entities.
        bundle = run_promote("She saw Sunlight through the window.")
        suppressed_reasons = {sc.reason for sc in bundle.suppressed}
        assert SuppressReason.LOW_ENTITYHOOD in suppressed_reasons

    def test_generic_verb_noise_is_suppressed_before_review(self):
        # Quote-initial contractions such as "Let's" currently normalize to
        # the verb lemma "let", which can survive recurrence with weak
        # possessive-like support. They should still be suppressed when no
        # real entity evidence ever develops around them.
        bundle = run_promote("\"Let's go home.\" \"Let us think about this later.\"")
        by_key = {candidate.cluster.normalized_key: candidate for candidate in bundle.suppressed}
        assert "let" in by_key
        assert by_key["let"].reason == SuppressReason.GENERIC_LEXICAL_NOISE

    def test_titled_name_is_not_suppressed_as_generic_verb_noise(self):
        # The generic-verb suppression rule must stay late and conservative.
        # A real titled character name that happens to look like a verb in
        # lowercase must still survive because the title is stronger evidence.
        bundle = run_promote("Captain Hope arrived. Captain Hope waved.")
        promoted_keys = {candidate.cluster.normalized_key for candidate in bundle.promoted}
        assert "hope" in promoted_keys

    def test_generic_component_overlap_is_suppressed_beneath_longer_compound(self):
        # When a generic-looking fragment appears only as a covered component
        # of a longer compound, the shorter fragment should not survive into
        # review on its own. This locks in the overlap-cleanup boundary for
        # phrases like "Old Man Hiroshi".
        bundle = run_promote("Old Man Hiroshi's shop smelled of dust.")
        suppressed = {candidate.cluster.normalized_key: candidate for candidate in bundle.suppressed}
        review_keys = {candidate.cluster.normalized_key for candidate in bundle.review_only}
        assert "old" in suppressed
        assert suppressed["old"].reason == SuppressReason.COMPONENT_OVERLAP_NOISE
        assert "old man" in suppressed
        assert suppressed["old man"].reason == SuppressReason.COMPONENT_OVERLAP_NOISE
        assert "old man hiroshi" in review_keys

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
        bundle = promote(pre, result.clusters, result.lexicon, attribution_records).bundle
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

    def test_weak_unresolved_recurring_cluster_is_suppressed(self):
        # A recurring bare-cap cluster with no title, possessive, attribution,
        # or class-specific evidence can accumulate enough scene-dispersion
        # score to enter review_only. Entityhood filtering must suppress it so
        # the unresolved bucket is reserved for plausible entities.
        text = "Still waited.\n\n---\n\nStill listened.\n\n---\n\nStill lingered."
        doc = parse("doc.md", text)
        pre = preprocess(doc)
        anchors = []
        start = 0
        while True:
            found = text.find("Still", start)
            if found == -1:
                break
            anchors.append(SpanAnchor(
                path="doc.md",
                span_ordinal=0,
                start_char=found,
                end_char=found + len("Still"),
            ))
            start = found + 1

        cluster = MentionCluster(
            normalized_key="still",
            surface_forms=["Still"],
            anchors=anchors,
            occurrence_count=len(anchors),
            title_support_count=0,
            possessive_support_count=0,
            location_support_count=0,
            linked_fields=[],
            linked_definitions=[],
            linked_seeds=[],
            cluster_id=stable_hash_id("doc.md", "still"),
        )

        bundle = promote(pre, [cluster], [], []).bundle
        suppressed_keys = {c.cluster.normalized_key for c in bundle.suppressed}
        review_keys = {c.cluster.normalized_key for c in bundle.review_only}
        assert "still" in suppressed_keys
        assert "still" not in review_keys
