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

from backend.nlp.classification.compound_shapes import compound_head, compound_parts
from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    EVENT_NOUNS,
    FACTION_SUFFIXES,
    PLACE_DESCRIPTOR_NOUNS,
    PLACE_OF_CONTEXT_NOUNS,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument

_NON_PERSON_DIRECTIONAL_HEADS: frozenset[str] = frozenset({"north", "south", "east", "west"})


def _looks_like_personal_compound(cluster: MentionCluster) -> bool:
    """Return True when a compound surface looks more like a person name.

    Multi-token compounds need some person-specific support so they do not all
    collapse into CHARACTER by default. This helper deliberately blocks obvious
    non-person heads such as institutional suffixes, place descriptors, and
    event nouns while allowing recurring personal-name compounds up to three
    tokens long.
    """
    parts = compound_parts(cluster)
    if len(parts) < 2 or len(parts) > 3:
        return False
    if any(not part.replace("'", "").isalpha() for part in parts):
        return False

    tail = compound_head(cluster)
    if tail is None:
        return False
    if tail in EVENT_NOUNS:
        return False
    if tail in PLACE_OF_CONTEXT_NOUNS or tail in PLACE_DESCRIPTOR_NOUNS:
        return False
    if tail in _NON_PERSON_DIRECTIONAL_HEADS:
        return False
    if tail.endswith(tuple(FACTION_SUFFIXES)):
        return False

    return True


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

    if _looks_like_personal_compound(cluster):
        score += 0.55
        reasons.append("appears as a personal-style multi-token compound")

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
