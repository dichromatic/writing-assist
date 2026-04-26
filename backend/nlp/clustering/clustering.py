"""
Evidence clustering - groups MentionCandidate records by normalised key.

All candidates that share the same normalised surface form are merged into a
single MentionCluster. The normalised key is pre-computed by the harvesting
stage (normalize_surface in shared.py), so clustering is a pure grouping
operation: no further string transformation is needed here.

.. code-block:: mermaid

    flowchart TD
        A[list of MentionCandidate] --> B[Group by normalized key]
        B --> C[For each group: merge anchors,\noccurrence count, surface forms,\nsupport flags]
        C --> D[Sort clusters by first-anchor\ndocument position]
        D --> E[list of MentionCluster\nlinked_fields/definitions/seeds empty]
"""

from __future__ import annotations

from backend.nlp.types import (
    MentionCandidate,
    MentionCluster,
)
from backend.nlp.harvesting.shared import stable_hash_id


def cluster_mentions(candidates: list[MentionCandidate]) -> list[MentionCluster]:
    """Group MentionCandidate records by normalised surface form.

    Candidates with the same normalised key (pre-computed by the harvesting
    stage) are merged into one MentionCluster. The cluster's occurrence_count
    is the total number of individual mentions, not the number of distinct
    surface forms.

    Clusters are returned sorted by the absolute document position of their
    first anchor, with the normalised key as a tiebreaker, so that the output
    order is deterministic regardless of the order candidates were passed in.

    The linked_fields, linked_definitions, and linked_seeds lists on every
    returned cluster are empty. Populate them by passing the clusters to
    link_clusters in linking.py.

    Args:
        candidates: MentionCandidate records from one or more harvesting passes,
            in any order.

    Returns:
        MentionCluster records sorted by first occurrence in the document.
        Returns an empty list when candidates is empty.
    """
    if not candidates:
        return []

    # Derive the document path from the first candidate. Within a single
    # pipeline run all candidates come from the same document.
    path = candidates[0].anchor.path

    groups: dict[str, list[MentionCandidate]] = {}
    for c in candidates:
        groups.setdefault(c.normalized, []).append(c)

    clusters: list[MentionCluster] = []
    for normalized_key, group in groups.items():
        # Collect distinct surface forms in first-seen order.
        seen: set[str] = set()
        surface_forms: list[str] = []
        for c in group:
            if c.surface not in seen:
                seen.add(c.surface)
                surface_forms.append(c.surface)

        clusters.append(MentionCluster(
            normalized_key=normalized_key,
            surface_forms=surface_forms,
            anchors=[c.anchor for c in group],
            occurrence_count=len(group),
            has_title_support=any(c.has_title_prefix for c in group),
            has_possessive_support=any(c.has_possessive for c in group),
            linked_fields=[],
            linked_definitions=[],
            linked_seeds=[],
            cluster_id=stable_hash_id(path, normalized_key),
        ))

    # Sort by the first anchor's document position so that the cluster list
    # order is determined by document order rather than dict insertion order.
    clusters.sort(key=lambda c: (c.anchors[0].start_char, c.normalized_key))
    return clusters
