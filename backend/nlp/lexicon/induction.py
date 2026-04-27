"""
Lexicon induction - derives BootstrappedLexiconEntry records from MentionClusters.

An entry is created for each non-possessive surface form of every cluster that
meets the induction threshold. Clusters below the threshold are discarded - they
do not have enough evidence to be treated as reliable patterns for second-pass
phrase matching.

.. code-block:: mermaid

    flowchart TD
        A[list of MentionCluster] --> B{Meets induction\nthreshold?}
        B -->|No: singleton, no title,\nno possessive| C[Discard]
        B -->|Yes| D[Assign category\nfrom cluster flags]
        D --> E[For each non-possessive\nsurface form]
        E --> F[Create BootstrappedLexiconEntry\nwith phrase and stable entry_id]
        F --> G[list of BootstrappedLexiconEntry]
"""

from __future__ import annotations

from backend.nlp.classification.arbitration import classify_cluster
from backend.nlp.types import BootstrappedLexiconEntry, LexiconCategory, MentionCluster
from backend.nlp.harvesting.shared import is_stopword, stable_hash_id


def _meets_induction_threshold(cluster: MentionCluster) -> bool:
    """Return True if the cluster has enough evidence to be inducted.

    A cluster is inducted when at least one of the following holds:
    - It has two or more occurrences (recurrence is strong evidence)
    - Any mention carried a title prefix (title is strong evidence)
    - Any mention was possessive (possessive implies personhood)

    A bare-capitalized singleton with none of these signals is too likely to be
    a sentence-initial common noun to be safely inducted.

    Args:
        cluster: A cluster from the clustering stage.

    Returns:
        True if the cluster meets the induction threshold.
    """
    return (
        cluster.occurrence_count >= 2
        or cluster.has_title_support
        or cluster.has_possessive_support
    )


def classify_clusters(
    clusters: list[MentionCluster],
    pre,
    attribution_records: list,
) -> dict[str, LexiconCategory]:
    """Classify clusters using the shared classification arbitration layer.

    Args:
        clusters: All clusters from the bootstrap result.
        pre: PreprocessedDocument context.
        attribution_records: Speaker attribution records from attribute_dialogue.

    Returns:
        Mapping from normalized_key to top-level LexiconCategory.
    """
    decisions = {
        cluster.normalized_key: classify_cluster(cluster, pre, attribution_records)
        for cluster in clusters
    }
    return {
        normalized_key: decision.winning_category
        for normalized_key, decision in decisions.items()
    }


def _rule_sources(cluster: MentionCluster) -> list[str]:
    """Derive rule source labels from cluster-level support flags.

    Args:
        cluster: A cluster from the clustering stage.

    Returns:
        List of rule source strings that contributed to this cluster.
    """
    sources: list[str] = ['bare_capitalized']
    if cluster.has_title_support:
        sources.append('title_prefix')
    if cluster.has_possessive_support:
        sources.append('possessive')
    return sources


def induce_lexicon(
    clusters: list[MentionCluster],
    path: str,
    induction_pass: int,
) -> list[BootstrappedLexiconEntry]:
    """Induce BootstrappedLexiconEntry records from clustered mention evidence.

    For each cluster that meets the induction threshold, one entry is created
    per unique non-possessive surface form. Possessive surface forms (ending in
    "'s") are stripped to their base form, because the base form will match
    both the bare name and the possessive context via the word-boundary rule in
    the Aho-Corasick matcher.

    Clusters whose normalized_key is a stopword are excluded regardless of
    occurrence count, since stopwords cannot name an entity.

    Args:
        clusters: MentionCluster records from the clustering stage, in any order.
        path: Document path, used to construct stable entry_ids.
        induction_pass: 0-based index of the convergence pass in which these
            entries are being inducted. Stored on the entry for audit purposes.

    Returns:
        BootstrappedLexiconEntry records ready for compilation into an
        Aho-Corasick automaton. May be empty if no clusters meet the threshold.
    """
    entries: list[BootstrappedLexiconEntry] = []

    for cluster in clusters:
        if is_stopword(cluster.normalized_key):
            continue
        if not _meets_induction_threshold(cluster):
            continue

        # Bootstrap-time induction has no attribution evidence yet. The
        # classifier therefore runs with an empty attribution set and only
        # uses the cluster's structural signals.
        category = classify_cluster(cluster=cluster, pre=None, attribution_records=[]).winning_category
        rule_srcs = _rule_sources(cluster)

        # Collect candidate phrases: for each surface form, use the base
        # (stripping any possessive suffix) so the automaton pattern covers
        # both the bare and possessive contexts.
        seen_phrases: set[str] = set()
        for surface in cluster.surface_forms:
            phrase = surface[:-2] if surface.endswith("'s") else surface
            if not phrase or phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            entries.append(BootstrappedLexiconEntry(
                phrase=phrase,
                normalized_phrase=cluster.normalized_key,
                category=category,
                anchors=list(cluster.anchors),
                occurrence_count=cluster.occurrence_count,
                archetypes_seen=['manuscript'],
                rule_sources=rule_srcs,
                induction_pass=induction_pass,
                entry_id=stable_hash_id(path, phrase),
            ))

    return entries
