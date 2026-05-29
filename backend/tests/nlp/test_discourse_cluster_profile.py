"""Tests for cluster-side discourse enrichment used by Stage 1 entityhood."""

from backend.nlp.clustering.clustering import cluster_mentions
from backend.nlp.discourse.cluster_profile import enrich_clusters_with_discourse
from backend.nlp.harvesting.manuscript import harvest_manuscript
from backend.nlp.parsing.markdown_parser import parse
from backend.nlp.parsing.preprocessing import preprocess


def _cluster_with_discourse(text: str, key: str):
    """Run harvest, cluster, and Stage 1 discourse enrichment for one key.

    Args:
        text: Manuscript text to process.
        key: Normalized cluster key to return.

    Returns:
        Discourse-enriched mention cluster for the requested key.
    """
    doc = parse("doc.md", text)
    pre = preprocess(doc)
    candidates = harvest_manuscript(pre)
    clusters = cluster_mentions(candidates)
    enrich_clusters_with_discourse(pre, clusters)
    return next(cluster for cluster in clusters if cluster.normalized_key == key)


def test_cluster_discourse_profile_distinguishes_quote_address_from_prose_support():
    # Stage 1 only helps entityhood if the cluster-side profile preserves the
    # same quote-vs-prose distinction the record-side discourse profile already
    # exposed. Losing the non-quote support count here would over-penalize real
    # names that are addressed in dialogue but also used in prose.
    aldous = _cluster_with_discourse(
        '"Aldous," Kohaku said. Aldous waited.',
        "aldous",
    )

    assert aldous.discourse_profile.in_quote_count == 1
    assert aldous.discourse_profile.non_quote_count == 1
    assert aldous.discourse_profile.quote_only is False
    assert aldous.discourse_profile.address_like_count == 1
    assert aldous.discourse_profile.one_token_utterance_count == 1


def test_cluster_discourse_profile_counts_one_token_quote_utterances():
    # One-token utterances are one of the two Stage 1 entityhood penalties.
    # This test locks the current quote-local interpretation so the earlier
    # migration keeps parity with the record-side signal and later rescue gate.
    aldous = _cluster_with_discourse(
        '"Aldous." "Aldous."',
        "aldous",
    )

    assert aldous.discourse_profile.in_quote_count == 2
    assert aldous.discourse_profile.non_quote_count == 0
    assert aldous.discourse_profile.quote_only is True
    assert aldous.discourse_profile.address_like_count == 0
    assert aldous.discourse_profile.one_token_utterance_count == 2
