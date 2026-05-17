"""
Group classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Suffix heuristic]
        A --> C[Membership context]
        A --> D[Leadership context]
        A --> E[Collective action verbs]
        B & C & D & E --> F[Group ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.scoring_builder import ScoringBuilder
from backend.nlp.classification.token_context import has_anchor_token_pattern
from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    FACTION_SUFFIXES,
    GROUP_COLLECTIVE_VERBS,
    GROUP_LEADERSHIP_NOUNS,
    GROUP_MEMBERSHIP_PREPOSITIONS,
    GROUP_MEMBERSHIP_VERBS,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument, Token


def _has_membership_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when the cluster appears in member-of style prose."""
    def _matches(tokens: list[Token], index: int) -> bool:
        if index >= 2:
            if (
                tokens[index - 2].text.lower() in GROUP_MEMBERSHIP_VERBS
                and tokens[index - 1].text.lower() in GROUP_MEMBERSHIP_PREPOSITIONS
            ):
                return True
        return False

    return has_anchor_token_pattern(cluster, pre, _matches)


def _has_leadership_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when leadership framing names the cluster as a collective."""
    def _matches(tokens: list[Token], index: int) -> bool:
        if index >= 2:
            if (
                tokens[index - 2].text.lower() in GROUP_LEADERSHIP_NOUNS
                and tokens[index - 1].text.lower() == "of"
            ):
                return True
        return False

    return has_anchor_token_pattern(cluster, pre, _matches)


def _has_collective_action_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby verbs describe group-like collective action."""
    def _matches(tokens: list[Token], index: int) -> bool:
        right_window = tokens[index + 1:index + 4]
        return any(token.text.lower() in GROUP_COLLECTIVE_VERBS for token in right_window)

    return has_anchor_token_pattern(cluster, pre, _matches)


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
    builder = ScoringBuilder(LexiconCategory.GROUP)

    if any(cluster.normalized_key.endswith(suffix) for suffix in FACTION_SUFFIXES):
        builder.set(0.75, "normalized key ends with a group-like suffix")

    if _has_membership_context(cluster, pre):
        builder.add(0.40, "appears in membership context")

    if _has_leadership_context(cluster, pre):
        builder.add(0.35, "appears in leadership framing")

    if _has_collective_action_context(cluster, pre):
        builder.add(0.35, "appears near a collective action verb")

    return builder.build()
