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

from backend.nlp.types import (
    BootstrappedLexiconEntry,
    LexiconCategory,
    MentionCluster,
)
from backend.nlp.harvesting.shared import FACTION_SUFFIXES, is_stopword, stable_hash_id


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


def _assign_category(
    cluster: MentionCluster,
    attr_keys: frozenset[str] = frozenset(),
) -> LexiconCategory:
    """Determine the best-effort lexicon category from cluster evidence.

    Priority order matters because signals can co-occur. A place name can
    appear in possessive form ("Tairngire's hills") and a character can appear
    after a locative preposition ("she saw herself in Yō"). The attribution
    tie-break resolves this: any cluster that has been attributed as a speaker
    is definitively a person, so locative context cannot override that.

    Args:
        cluster: A cluster from the clustering stage.
        attr_keys: Normalized keys of all clusters that have been attributed as
            dialogue speakers. When called during bootstrapping (before
            attribution), pass the default empty set - the category will be
            corrected later via classify_clusters.

    Returns:
        The most specific LexiconCategory determinable from cluster evidence,
        or UNRESOLVED when no reliable signal is present.
    """
    # A title prefix ("Captain", "Lord") is unambiguous evidence of a person.
    if cluster.has_title_support:
        return LexiconCategory.CHARACTER

    # Location beats possessive only when the cluster has no attribution
    # evidence. An attributed cluster spoke in the text and is definitively a
    # person even if they also appear after a locative preposition.
    if cluster.has_location_support and cluster.normalized_key not in attr_keys:
        return LexiconCategory.PLACE

    # Faction/organisation suffixes are recognisable by the normalized key
    # alone - no candidate-level signal is needed.
    if any(cluster.normalized_key.endswith(suffix) for suffix in FACTION_SUFFIXES):
        return LexiconCategory.FACTION

    # Possessive form implies the entity is treated as a person by the author.
    if cluster.has_possessive_support:
        return LexiconCategory.CHARACTER

    return LexiconCategory.UNRESOLVED


def classify_clusters(
    clusters: list[MentionCluster],
    attribution_records: list,
) -> dict[str, LexiconCategory]:
    """Classify clusters using attribution evidence that was not available
    at bootstrap time.

    This is called after attribute_dialogue has run so that the attribution
    tie-break in _assign_category can resolve ambiguous cases where a cluster
    has both locative and possessive support (e.g. a character who appears
    after "in" in a figurative construction).

    Args:
        clusters: All clusters from the bootstrap result.
        attribution_records: Speaker attribution records from attribute_dialogue.
            Each record must have a speaker_key attribute.

    Returns:
        Mapping from normalized_key to corrected LexiconCategory.
    """
    attr_keys: frozenset[str] = frozenset(r.speaker_key for r in attribution_records)
    return {c.normalized_key: _assign_category(c, attr_keys) for c in clusters}


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

        category = _assign_category(cluster)
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
