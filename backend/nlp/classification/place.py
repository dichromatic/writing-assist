"""
Place classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Locative context]
        A --> C[Speaker veto]
        B & C --> D[Place ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.compound_shapes import compound_head
from backend.nlp.classification.token_context import iter_anchor_token_starts
from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    DEMONYM_SUFFIXES,
    PLACE_DESCRIPTOR_NOUNS,
    PLACE_OF_CONTEXT_NOUNS,
    PLACE_POSSESSIVE_CONTEXT_NOUNS,
    PLACE_RESIDENT_NOUNS,
    STRONG_LOCATIVE_PREPOSITIONS,
    WEAK_LOCATIVE_PREPOSITIONS,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument

# Compass directions are treated as a closed structural set rather than a
# coverage lexicon. They behave like place heads in compounds such as
# "Polar North", and WordNet expansion would add more ambiguity than value.
_COMPASS_PLACE_HEADS: frozenset[str] = frozenset({"north", "south", "east", "west"})


def _is_capitalized_word(text: str) -> bool:
    """Return True when text begins with an uppercase alphabetic character."""
    return bool(text) and text[0].isalpha() and text[0].isupper()


def _place_descriptor_support(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when local context names the cluster as a geographic place."""
    for tokens, index in iter_anchor_token_starts(cluster, pre):
        if index >= 2:
            if (
                tokens[index - 1].text.lower() == "of"
                and tokens[index - 2].text.lower() in PLACE_OF_CONTEXT_NOUNS
            ):
                return True

        if index + 2 < len(tokens):
            if (
                tokens[index + 1].text == ","
                and tokens[index + 2].text.lower() in {"the", "a", "an"}
                and index + 3 < len(tokens)
                and tokens[index + 3].text.lower() in PLACE_DESCRIPTOR_NOUNS
            ):
                return True

        if index + 1 < len(tokens) and tokens[index + 1].text.lower() in PLACE_DESCRIPTOR_NOUNS:
            return True

    return False


def _possessive_place_support(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when possessive syntax clearly frames the cluster as a place."""
    if pre is None or not cluster.has_possessive_support:
        return False

    for tokens, index in iter_anchor_token_starts(cluster, pre):
        next_token = tokens[index + 1].text.lower() if index + 1 < len(tokens) else ""
        next_next = tokens[index + 2].text.lower() if index + 2 < len(tokens) else ""

        if next_token in {"'s", "'"} and next_next in PLACE_POSSESSIVE_CONTEXT_NOUNS:
            return True

    return False


def _resident_place_support(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when resident nouns frame the cluster as a place."""
    for tokens, index in iter_anchor_token_starts(cluster, pre):
        if index + 1 < len(tokens) and tokens[index + 1].text.lower() in PLACE_RESIDENT_NOUNS:
            return True

    return False


def _locative_strength(cluster: MentionCluster, pre: PreprocessedDocument | None) -> tuple[float, list[str], list[str]]:
    """Refine harvest-time location flags using neighboring token context."""
    if pre is None or not cluster.has_location_support:
        return 0.0, [], []

    strong_hits = 0
    weak_hits = 0
    weak_compound_hits = 0

    for tokens, index in iter_anchor_token_starts(cluster, pre):
        if index == 0:
            continue

        preceding = tokens[index - 1].text.lower()
        if preceding in {"the", "a", "an"} and index >= 2:
            preceding = tokens[index - 2].text.lower()
        following = tokens[index + 1].text if index + 1 < len(tokens) else ""

        if preceding in STRONG_LOCATIVE_PREPOSITIONS:
            strong_hits += 1
        elif preceding in WEAK_LOCATIVE_PREPOSITIONS:
            weak_hits += 1
            if _is_capitalized_word(following):
                weak_compound_hits += 1

    reasons: list[str] = []
    vetoes: list[str] = []

    if strong_hits:
        reasons.append("appears after a strong locative preposition")
        return 0.60, reasons, vetoes

    if weak_hits:
        reasons.append("appears after a weak locative preposition")
        if weak_compound_hits == weak_hits:
            vetoes.append("weak locative context only appears in a capitalized compound")
            return 0.0, reasons, vetoes
        return 0.25, reasons, vetoes

    return 0.0, reasons, vetoes


def score_place_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
    attributed_speakers: frozenset[str],
) -> ClassEvidence:
    """Score how strongly a cluster behaves like a place.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context. Reserved for future use.
        attributed_speakers: Normalized keys that were attributed as speakers.

    Returns:
        Place evidence for the cluster.
    """
    score = 0.0
    reasons: list[str] = []
    vetoes: list[str] = []

    locative_score, locative_reasons, locative_vetoes = _locative_strength(cluster, pre)
    score += locative_score
    reasons.extend(locative_reasons)
    vetoes.extend(locative_vetoes)

    if _place_descriptor_support(cluster, pre):
        score += 0.60
        reasons.append("appears with a geographic descriptor")

    if _possessive_place_support(cluster, pre):
        score += 0.60
        reasons.append("appears in possessive place context")

    if _resident_place_support(cluster, pre):
        score += 0.60
        reasons.append("appears with resident or civic framing")

    head = compound_head(cluster)
    if head in PLACE_DESCRIPTOR_NOUNS:
        score += 0.65
        reasons.append("compound head is a place-like descriptor")
    elif head in _COMPASS_PLACE_HEADS:
        score += 0.65
        reasons.append("compound head is a directional place noun")

    if (
        cluster.normalized_key.endswith(tuple(DEMONYM_SUFFIXES))
        and score < 0.60
    ):
        vetoes.append("demonym-like form without stronger place evidence")
        score = min(score, 0.15)

    if cluster.normalized_key in attributed_speakers:
        vetoes.append("attributed speaker evidence blocks place resolution")
        score = 0.0

    return ClassEvidence(
        category=LexiconCategory.PLACE,
        score=min(score, 1.0),
        reasons=reasons,
        vetoes=vetoes,
    )
