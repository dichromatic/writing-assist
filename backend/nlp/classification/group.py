"""
Group classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Suffix heuristic]
        B --> C[Group ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import FACTION_SUFFIXES
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def score_group_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Score how strongly a cluster behaves like a collective entity.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context. Reserved for future use.

    Returns:
        Group evidence for the cluster.
    """
    del pre

    reasons: list[str] = []
    score = 0.0

    if any(cluster.normalized_key.endswith(suffix) for suffix in FACTION_SUFFIXES):
        score = 0.75
        reasons.append("normalized key ends with a group-like suffix")

    return ClassEvidence(
        category=LexiconCategory.GROUP,
        score=score,
        reasons=reasons,
        vetoes=[],
    )
