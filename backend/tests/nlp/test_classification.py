"""
Tests for backend/nlp/classification/*.

These tests lock in the top-level classification contract and the arbitration
rules that separate character/group/place decisions from the rest of the NLP
pipeline.
"""

from backend.nlp.classification.arbitration import (
    classify_cluster,
    classify_clusters,
)
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.clustering.linking import link_clusters
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.types import LexiconCategory, MentionCluster, stable_hash_id


def harvest_and_cluster(text: str, path: str = "doc.md"):
    doc = parse(path, text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    link_clusters(clusters, [], [], [])
    return pre, clusters


def make_cluster(
    normalized_key: str,
    *,
    occurrence_count: int = 1,
    has_title_support: bool = False,
    has_possessive_support: bool = False,
    has_location_support: bool = False,
) -> MentionCluster:
    """Build a minimal cluster for classification-only tests."""
    return MentionCluster(
        normalized_key=normalized_key,
        surface_forms=[normalized_key.title()],
        anchors=[],
        occurrence_count=occurrence_count,
        has_title_support=has_title_support,
        has_possessive_support=has_possessive_support,
        has_location_support=has_location_support,
        linked_fields=[],
        linked_definitions=[],
        linked_seeds=[],
        cluster_id=stable_hash_id("doc.md", normalized_key),
    )


class TestClassification:
    def test_titled_cluster_resolves_character(self):
        # A titled entity is the clearest deterministic person signal.
        pre, clusters = harvest_and_cluster("She met Captain Aldous by the gate.")
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, [])
        assert decision.winning_category == LexiconCategory.CHARACTER
        assert decision.resolved is True

    def test_locative_cluster_resolves_place_without_attribution(self):
        # A capitalized name that appears after a locative preposition should
        # resolve as PLACE when no stronger person evidence exists.
        pre, clusters = harvest_and_cluster("She arrived in Tairngire. Tairngire glowed.")
        tairngire = next(c for c in clusters if c.normalized_key == "tairngire")
        decision = classify_cluster(tairngire, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_attribution_beats_place_evidence(self):
        # If a cluster is attributed as a speaker, personhood outranks locative
        # context from some other occurrence.
        text = 'She stood in Aldous Hall. "Go now," Aldous said.'
        pre, clusters = harvest_and_cluster(text)
        records = attribute_dialogue(pre, clusters)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, records)
        assert decision.winning_category == LexiconCategory.CHARACTER
        assert decision.resolved is True

    def test_group_suffix_resolves_group(self):
        # Institutional suffixes should resolve to the collective group class,
        # not to the legacy faction-specific category.
        pre, clusters = harvest_and_cluster(
            "The Norre Institute closed its gates. The Institute remained silent."
        )
        institute = next(c for c in clusters if c.normalized_key == "institute")
        decision = classify_cluster(institute, pre, [])
        assert decision.winning_category == LexiconCategory.GROUP
        assert decision.resolved is True

    def test_place_descriptor_support_resolves_place(self):
        # A geographic descriptor around a capitalized name should be enough
        # to resolve placehood even without a preceding locative preposition.
        pre, clusters = harvest_and_cluster("The city of Sidhe slept beneath the fog.")
        sidhe = next(c for c in clusters if c.normalized_key == "sidhe")
        decision = classify_cluster(sidhe, pre, [])
        assert decision.winning_category == LexiconCategory.PLACE
        assert decision.resolved is True

    def test_possessive_only_cluster_remains_unresolved(self):
        # Possessive form alone is evidence of entityhood but not enough to
        # choose between person, place, object, or concept.
        pre, clusters = harvest_and_cluster("Aldous's sword was missing.")
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        decision = classify_cluster(aldous, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is True

    def test_recurring_bare_cluster_is_weak_entityhood(self):
        # Recurrence alone is not enough to call a cluster a trustworthy entity.
        # Weak recurring bare-cap clusters should remain visible to diagnostics
        # as unresolved, but they must not be treated as accepted entityhood.
        cluster = make_cluster("still", occurrence_count=4)
        decision = classify_cluster(cluster, None, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_weak_locative_compound_does_not_resolve_place(self):
        # A capitalized adjective inside an abstract compound such as
        # "Cosmic Time" must not resolve as a place just because it follows a
        # weak path preposition.
        pre, clusters = harvest_and_cluster("She fell through Cosmic Time.")
        cosmic = next(c for c in clusters if c.normalized_key == "cosmic")
        decision = classify_cluster(cosmic, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_demonym_like_cluster_after_weak_locative_stays_unresolved(self):
        # Demonym or adjectival forms such as "Lunarian" should not resolve as
        # places from a weak preposition alone.
        text = (
            "Tea was gathered by Lunarian druids. "
            "Later, Lunarian chants faded into the valley."
        )
        pre, clusters = harvest_and_cluster(text)
        lunarian = next(c for c in clusters if c.normalized_key == "lunarian")
        decision = classify_cluster(lunarian, pre, [])
        assert decision.winning_category == LexiconCategory.UNRESOLVED
        assert decision.resolved is False
        assert decision.entityhood.accepted is False

    def test_bulk_classification_returns_mapping_by_cluster_key(self):
        pre, clusters = harvest_and_cluster("She met Captain Aldous in Tairngire.")
        decisions = classify_clusters(clusters, pre, [])
        assert set(decisions) == {c.normalized_key for c in clusters}
