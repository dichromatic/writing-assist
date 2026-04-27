"""
Character classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Title support]
        A --> C[Possessive support]
        A --> D[Dialogue attribution]
        B & C & D --> E[Character ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def score_character_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
    attributed_speakers: frozenset[str],
) -> ClassEvidence:
    """Score how strongly a cluster behaves like a singular actor.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context. Reserved for future use.
        attributed_speakers: Normalized keys that were attributed as speakers.

    Returns:
        Character evidence for the cluster.
    """
    del pre

    score = 0.0
    reasons: list[str] = []
    vetoes: list[str] = []

    if cluster.normalized_key in attributed_speakers:
        score += 0.80
        reasons.append("attributed as a dialogue speaker")

    if cluster.has_title_support:
        score += 0.70
        reasons.append("appears with a title prefix")

    if cluster.has_possessive_support:
        score += 0.35
        reasons.append("appears in possessive form")

    if cluster.occurrence_count >= 2:
        score += 0.10
        reasons.append("recurs across the document")

    if cluster.has_location_support and cluster.normalized_key not in attributed_speakers:
        vetoes.append("has locative context without attribution support")

    return ClassEvidence(
        category=LexiconCategory.CHARACTER,
        score=min(score, 1.0),
        reasons=reasons,
        vetoes=vetoes,
    )
