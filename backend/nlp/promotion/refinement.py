"""
Post-entityhood promotion refinement - apply deterministic unresolved suppression.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster list] --> B[Look up ClassificationDecision]
        B --> C{Winning category unresolved?}
        C -->|No| D[Skip]
        C -->|Yes| E[Evaluate discourse suppression policy]
        E -->|Matched| F[Record SuppressReason and detail]
        E -->|No match| D
        D & F --> G[Refined suppression map]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassificationDecision
from backend.nlp.discourse.filtering import (
    discourse_suppression_detail,
    unresolved_discourse_suppression_reason,
)
from backend.nlp.types import MentionCluster, SuppressReason


def build_post_entityhood_unresolved_suppressions(
    clusters: list[MentionCluster],
    classifications: dict[str, ClassificationDecision],
) -> dict[str, tuple[SuppressReason, str]]:
    """Build a structural suppression map for unresolved discourse-shaped clusters.

    This refinement step runs after broad entityhood classification but before
    promotion bucket routing. It lets entityhood stay a plausibility gate while
    still suppressing quote-only unresolved junk earlier than rescue.

    Args:
        clusters: Mention clusters already enriched with discourse evidence.
        classifications: Deterministic classification decisions by cluster key.

    Returns:
        Mapping from normalized key to a suppression reason and human-readable
        detail string for clusters that should be suppressed structurally.
    """
    refined: dict[str, tuple[SuppressReason, str]] = {}
    for cluster in clusters:
        decision = classifications[cluster.normalized_key]
        reason = unresolved_discourse_suppression_reason(cluster, decision)
        if reason is None:
            continue
        refined[cluster.normalized_key] = (
            reason,
            discourse_suppression_detail(cluster, reason),
        )
    return refined
