"""
Cluster linking - attaches StructuredFieldCandidate, DefinitionCandidate,
and SectionSummarySeed records to MentionClusters whose key they reference.

Linking is a separate pass from clustering so that the clustering module stays
a pure grouping operation with no knowledge of other candidate types. Both
modules operate on already-clustered data from clustering.py.

.. code-block:: mermaid

    flowchart TD
        A[list of MentionCluster] --> D[Build key-to-cluster index]
        B[list of StructuredFieldCandidate] --> E[Normalize field value\nand match to cluster key]
        C[list of DefinitionCandidate] --> F[Normalize term\nand match to cluster key]
        G[list of SectionSummarySeed] --> H[Extract words from heading\nand sentences, match keys]
        D & E --> I[Append to cluster.linked_fields]
        D & F --> J[Append to cluster.linked_definitions]
        D & H --> K[Append to cluster.linked_seeds]
        I & J & K --> L[Mutated MentionCluster list]
"""

from __future__ import annotations

from backend.nlp.types import (
    DefinitionCandidate,
    MentionCluster,
    SectionSummarySeed,
    StructuredFieldCandidate,
)


def _normalize_for_linking(text: str) -> str:
    """Lowercase and strip whitespace for cross-type key matching.

    This is intentionally simpler than normalize_surface: linking candidates
    (field values, definition terms) do not have title prefixes or possessives
    to strip. Plain lowercase normalisation is sufficient and predictable.

    Args:
        text: Raw text from a field value or definition term.

    Returns:
        Lowercased, stripped text.
    """
    return text.lower().strip()


def _seed_words(seed: SectionSummarySeed) -> set[str]:
    """Extract a set of lowercase words from a seed's heading and sentences.

    Strips common sentence-boundary punctuation so that "aldous," and "aldous"
    both match the cluster key "aldous".

    Args:
        seed: A section summary seed from the harvesting stage.

    Returns:
        Set of lowercase, punctuation-stripped words from the seed text.
    """
    parts: list[str] = []
    if seed.heading_text:
        parts.append(seed.heading_text)
    parts.extend(seed.key_sentences)
    words: set[str] = set()
    for part in parts:
        for word in part.split():
            words.add(word.lower().strip('.,!?;:\'"()'))
    return words


def link_clusters(
    clusters: list[MentionCluster],
    fields: list[StructuredFieldCandidate],
    definitions: list[DefinitionCandidate],
    seeds: list[SectionSummarySeed],
) -> list[MentionCluster]:
    """Attach related candidates and seeds to the clusters that reference them.

    Field candidates are matched by normalising the field value and looking up
    the corresponding cluster. Definition candidates are matched by normalising
    the term. Seeds are matched by checking whether any word in the seed text
    matches a cluster's normalised key.

    Each input list may be empty; empty inputs produce no links for that type.
    The clusters list is mutated in place and also returned for chaining.

    Args:
        clusters: MentionCluster records from cluster_mentions, whose link
            lists are expected to be empty on entry.
        fields: StructuredFieldCandidate records from the harvesting stage.
        definitions: DefinitionCandidate records from the harvesting stage.
        seeds: SectionSummarySeed records from the harvesting stage.

    Returns:
        The same clusters list, with linked_fields, linked_definitions, and
        linked_seeds populated where matches were found.
    """
    cluster_by_key: dict[str, MentionCluster] = {
        c.normalized_key: c for c in clusters
    }

    for field in fields:
        key = _normalize_for_linking(field.value)
        if key in cluster_by_key:
            cluster_by_key[key].linked_fields.append(field)

    for defn in definitions:
        key = _normalize_for_linking(defn.term)
        if key in cluster_by_key:
            cluster_by_key[key].linked_definitions.append(defn)

    for seed in seeds:
        words = _seed_words(seed)
        for cluster in clusters:
            if cluster.normalized_key in words:
                cluster.linked_seeds.append(seed)

    return clusters
