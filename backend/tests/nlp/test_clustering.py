"""
Tests for backend/nlp/clustering/clustering.py and
        backend/nlp/clustering/linking.py.

Each test encodes a non-obvious decision. The comments explain what decision
is being locked in and why its breakage would be invisible.
"""

import pytest

from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.clustering.linking import link_clusters
from backend.nlp.types import (
    DefinitionCandidate,
    SectionAnchor,
    SectionSummarySeed,
    SpanAnchor,
    StructuredFieldCandidate,
    stable_hash_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pipeline(text: str, path: str = "doc.md"):
    """Full pipeline: parse -> preprocess -> harvest -> cluster."""
    doc = parse(path, text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    return cluster_mentions(candidates)


def make_field(label: str, value: str, path: str = "doc.md") -> StructuredFieldCandidate:
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(value))
    return StructuredFieldCandidate(
        label=label,
        value=value,
        anchor=anchor,
        candidate_id=stable_hash_id(path, "0", label),
    )


def make_definition(term: str, definition_text: str, path: str = "doc.md") -> DefinitionCandidate:
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(term))
    return DefinitionCandidate(
        term=term,
        definition_text=definition_text,
        anchor=anchor,
        candidate_id=stable_hash_id(path, "0", term),
    )


def make_seed(heading: str | None, sentences: list[str], path: str = "doc.md") -> SectionSummarySeed:
    anchor = SectionAnchor(path=path, section_index=0)
    return SectionSummarySeed(
        anchor=anchor,
        heading_text=heading,
        key_sentences=sentences,
        candidate_id=stable_hash_id(path, "0"),
    )


# ---------------------------------------------------------------------------
# Clustering: grouping
# ---------------------------------------------------------------------------

class TestClustering:
    def test_same_normalized_key_clusters_together(self):
        # The whole point of clustering: "Aldous", "Aldous's", and
        # "Captain Aldous" all normalize to "aldous" and must produce exactly
        # one cluster. If any form produced a separate cluster, the same
        # character would appear as multiple entities in the output.
        text = "Captain Aldous arrived. Aldous sat. Aldous's sword fell."
        clusters = pipeline(text)
        aldous_clusters = [c for c in clusters if c.normalized_key == "aldous"]
        assert len(aldous_clusters) == 1

    def test_occurrence_count_is_total_mentions_not_distinct_surfaces(self):
        # occurrence_count must be the total number of candidate records merged,
        # not the number of distinct surface forms. The promotion stage uses
        # occurrence_count as a weight; counting distinct surfaces instead would
        # report a character mentioned 20 times as count=1 if they always appear
        # with the same form.
        text = "Aldous came. Aldous left. Aldous returned."
        clusters = pipeline(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert aldous.occurrence_count == 3

    def test_all_anchors_preserved_in_cluster(self):
        # Every individual mention's anchor must survive merging. Dropping any
        # anchor would silently prevent retrieval from finding that passage when
        # evidence is looked up for a cluster.
        text = "Aldous came. She greeted Aldous warmly. Aldous left."
        clusters = pipeline(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert len(aldous.anchors) == 3

    def test_distinct_surface_forms_deduplicated(self):
        # surface_forms must list each unique form only once. Duplicates would
        # cause the lexicon bootstrapper to create duplicate entries for the
        # same surface, inflating the lexicon with redundant patterns.
        text = "Aldous came. Aldous left. Captain Aldous spoke."
        clusters = pipeline(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert len(aldous.surface_forms) == len(set(aldous.surface_forms))

    def test_has_title_support_is_true_if_any_candidate_has_title(self):
        # has_title_support is an OR across all candidates in the cluster,
        # not just the first. A cluster where one occurrence is "Captain Aldous"
        # and the rest are bare "Aldous" must still have has_title_support=True.
        # Checking only the first candidate would miss title evidence when the
        # titled form appears mid-document.
        text = "Aldous walked in. She met Captain Aldous."
        clusters = pipeline(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert aldous.has_title_support is True

    def test_has_possessive_support_is_true_if_any_candidate_has_possessive(self):
        # Same OR-across-all-candidates logic for possessives. A cluster where
        # only one mention is possessive must still expose the possessive signal.
        text = "Aldous's sword fell. Aldous retrieved it."
        clusters = pipeline(text)
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert aldous.has_possessive_support is True

    def test_cluster_order_is_by_first_document_position(self):
        # Clusters must be sorted by the position of their first anchor in the
        # document, not by dict insertion order (which depends on candidate
        # processing order). A different order on each run would make stored
        # output non-reproducible and tests flaky.
        text = "Aldous came. Rhea followed. Aldous turned. Rhea smiled."
        clusters = pipeline(text)
        positions = [c.anchors[0].start_char for c in clusters]
        assert positions == sorted(positions)

    def test_clustering_is_deterministic_across_input_order(self):
        # The same set of candidates, presented in different orders, must
        # produce the same cluster_ids. If the cluster_id depended on processing
        # order rather than on the normalized_key, deduplication across runs
        # would silently fail.
        text = "Aldous came. Rhea followed."
        doc = parse("doc.md", text)
        pre = preprocess(doc)
        candidates = harvest_manuscript(pre)

        clusters_a = cluster_mentions(candidates)
        clusters_b = cluster_mentions(list(reversed(candidates)))
        assert {c.cluster_id for c in clusters_a} == {c.cluster_id for c in clusters_b}

    def test_cluster_id_is_stable_across_runs(self):
        # cluster_id must be the same every time the same document is clustered.
        # A non-deterministic ID (e.g. using Python's id()) would treat every
        # pipeline run as producing new records, making downstream deduplication
        # impossible.
        text = "Captain Aldous arrived."
        clusters_1 = pipeline(text)
        clusters_2 = pipeline(text)
        assert [c.cluster_id for c in clusters_1] == [c.cluster_id for c in clusters_2]

    def test_empty_candidates_returns_empty_list(self):
        # The function must return [] for empty input rather than crashing.
        # A crash on empty input would halt the pipeline for documents that
        # happen to produce no harvestable candidates.
        assert cluster_mentions([]) == []

    def test_new_cluster_has_empty_link_lists(self):
        # Freshly built clusters must have empty linked_fields, linked_definitions,
        # and linked_seeds. Populated lists would indicate stale state leaking
        # between pipeline runs, which could corrupt promotion-stage decisions.
        text = "Aldous arrived."
        clusters = pipeline(text)
        for c in clusters:
            assert c.linked_fields == []
            assert c.linked_definitions == []
            assert c.linked_seeds == []


# ---------------------------------------------------------------------------
# Linking: field candidates
# ---------------------------------------------------------------------------

class TestLinking:
    def test_field_linked_to_matching_cluster(self):
        # A StructuredFieldCandidate whose normalised value matches a cluster
        # key must appear in that cluster's linked_fields. This is the basic
        # linking contract.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        field = make_field("Alias", "Aldous")
        link_clusters(clusters, fields=[field], definitions=[], seeds=[])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert field in aldous.linked_fields

    def test_field_anchor_preserved_after_linking(self):
        # The linked field must be the same object (not a copy without anchor).
        # If linking stripped the anchor, the field's source provenance would
        # be silently lost and could not be retrieved later.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        field = make_field("Alias", "Aldous")
        original_anchor = field.anchor
        link_clusters(clusters, fields=[field], definitions=[], seeds=[])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert aldous.linked_fields[0].anchor is original_anchor

    def test_unmatched_field_not_linked_to_any_cluster(self):
        # A field whose normalised value does not match any cluster key must
        # not be attached to any cluster. Without this check, every field
        # would be linked everywhere, corrupting the associations used by
        # the promotion stage.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        field = make_field("Alias", "Rhea")  # "rhea" does not match "aldous"
        link_clusters(clusters, fields=[field], definitions=[], seeds=[])
        for c in clusters:
            assert field not in c.linked_fields

    def test_definition_linked_to_matching_cluster(self):
        # A DefinitionCandidate whose normalised term matches a cluster key
        # must appear in that cluster's linked_definitions.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        defn = make_definition("Aldous", "A seasoned navigator.")
        link_clusters(clusters, fields=[], definitions=[defn], seeds=[])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert defn in aldous.linked_definitions

    def test_seed_linked_when_cluster_key_appears_in_heading(self):
        # A SectionSummarySeed must be linked to a cluster when the cluster's
        # normalised key appears as a word in the seed's heading_text. This
        # enables downstream stages to find the section most associated with
        # a given entity.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        seed = make_seed(heading="Aldous and the Storm", sentences=[])
        link_clusters(clusters, fields=[], definitions=[], seeds=[seed])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert seed in aldous.linked_seeds

    def test_seed_linked_when_cluster_key_appears_in_sentence(self):
        # Seeds are also matched by words in key_sentences, not only headings.
        # A cluster key appearing mid-sentence in a seed should link that seed
        # to the cluster.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        seed = make_seed(heading=None, sentences=["The journey of Aldous was long."])
        link_clusters(clusters, fields=[], definitions=[], seeds=[seed])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert seed in aldous.linked_seeds

    def test_seed_not_linked_when_key_absent_from_seed_text(self):
        # A seed that does not mention the cluster's normalised key must not
        # be linked to that cluster. Spurious links would associate unrelated
        # sections with entities that never appear in them.
        text = "She greeted Aldous warmly. Aldous smiled."
        clusters = pipeline(text)
        seed = make_seed(heading="Unrelated chapter", sentences=["Nothing relevant here."])
        link_clusters(clusters, fields=[], definitions=[], seeds=[seed])
        aldous = next(c for c in clusters if c.normalized_key == "aldous")
        assert seed not in aldous.linked_seeds

    def test_link_clusters_returns_same_list(self):
        # link_clusters mutates the clusters in place and returns the same list
        # object. A caller that ignores the return value and uses the original
        # list must still see the links. If the function returned a new list,
        # callers that assigned the result to a new variable would be fine, but
        # callers that relied on in-place mutation (the more natural pattern)
        # would silently get unlinked clusters.
        text = "Aldous arrived."
        clusters = pipeline(text)
        original_list = clusters
        result = link_clusters(clusters, fields=[], definitions=[], seeds=[])
        assert result is original_list
