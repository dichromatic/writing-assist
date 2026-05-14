"""
Entityhood scoring for clustered mention candidates.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Structural evidence]
        A --> C[Linked reference evidence]
        A --> D[Recurrence evidence]
        B & C & D --> E[EntityhoodDecision]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence, EntityhoodDecision
from backend.nlp.harvesting.shared import TITLE_PREFIXES_LOWER
from backend.nlp.types import LexiconCategory, MentionCluster

_ENTITYHOOD_THRESHOLD = 0.55
def assess_entityhood(
    cluster: MentionCluster,
    evidence_by_category: dict[LexiconCategory, ClassEvidence],
    title_prefixes_lower: frozenset[str] = TITLE_PREFIXES_LOWER,
) -> EntityhoodDecision:
    """Score whether a cluster is plausible enough to survive as an entity.

    The score deliberately answers a narrower question than top-level
    classification: not "what is this?" but "is this likely enough to be a
    real named thing that review should keep it around?"

    Args:
        cluster: Cluster being evaluated.
        evidence_by_category: Per-category evidence already computed for the
            cluster during classification arbitration.

    Returns:
        An EntityhoodDecision with score, acceptance flag, and trace reasons.
    """
    score = 0.0
    reasons: list[str] = []
    weaknesses: list[str] = []

    if cluster.has_title_support:
        score += 0.75
        reasons.append("appears with a title prefix")
    elif cluster.normalized_key in title_prefixes_lower:
        score += 0.55
        reasons.append("is used as a bare title reference")

    if cluster.has_possessive_support:
        score += 0.55
        reasons.append("appears in possessive form")

    place_score = evidence_by_category[LexiconCategory.PLACE].score
    if place_score >= 0.60:
        score += 0.55
        reasons.append("has strong place-context evidence")
    elif place_score >= 0.25:
        score += 0.20
        reasons.append("has weak place-context evidence")

    character_score = evidence_by_category[LexiconCategory.CHARACTER].score
    if character_score >= 0.80:
        score += 0.80
        reasons.append("is attributed as an acting speaker")

    for category, label in (
        (LexiconCategory.GROUP, "group"),
        (LexiconCategory.EVENT, "event"),
        (LexiconCategory.CONCEPT, "concept"),
    ):
        if evidence_by_category[category].score >= 0.60:
            score += 0.45
            reasons.append(f"has strong {label}-classification evidence")

    if cluster.linked_fields:
        score += 0.45
        reasons.append("is referenced by structured fields")

    if cluster.linked_definitions:
        score += 0.45
        reasons.append("is referenced by definition-style notes")

    if cluster.linked_seeds:
        score += 0.20
        reasons.append("appears in section summary seeds")

    if cluster.occurrence_count >= 2:
        score += 0.25
        reasons.append("recurs across the document")
    else:
        weaknesses.append("appears only once")

    # A recurring multi-token compound carries more naming weight than a
    # recurring single word. Two adjacent capitalized words appearing 2+ times
    # is strong evidence of a deliberate named reference, independent of whether
    # the classification layer could resolve its category.
    if len(cluster.normalized_key.split()) >= 2 and cluster.occurrence_count >= 2:
        score += 0.30
        reasons.append("recurs as a multi-token compound")

    if (
        not cluster.has_title_support
        and not cluster.has_possessive_support
        and not cluster.linked_fields
        and not cluster.linked_definitions
        and not cluster.linked_seeds
        and place_score < 0.25
        and character_score < 0.80
    ):
        weaknesses.append("has no structural support beyond capitalization")

    return EntityhoodDecision(
        score=min(score, 1.0),
        accepted=score >= _ENTITYHOOD_THRESHOLD,
        reasons=reasons,
        weaknesses=weaknesses,
    )
