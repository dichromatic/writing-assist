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

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    FACTION_SUFFIXES,
    GROUP_COLLECTIVE_VERBS,
    GROUP_LEADERSHIP_NOUNS,
    GROUP_MEMBERSHIP_PREPOSITIONS,
    GROUP_MEMBERSHIP_VERBS,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def _token_for_anchor(cluster: MentionCluster, pre: PreprocessedDocument | None):
    """Yield the span token list and token index for each anchor in the cluster."""
    if pre is None:
        return

    for anchor in cluster.anchors:
        tokens = pre.tokens_by_span.get(anchor.span_ordinal, [])
        for index, token in enumerate(tokens):
            if token.start_char == anchor.start_char and token.end_char == anchor.end_char:
                yield tokens, index
                break


def _has_membership_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when the cluster appears in member-of style prose."""
    for tokens, index in _token_for_anchor(cluster, pre):
        if index >= 2:
            if (
                tokens[index - 2].text.lower() in GROUP_MEMBERSHIP_VERBS
                and tokens[index - 1].text.lower() in GROUP_MEMBERSHIP_PREPOSITIONS
            ):
                return True
    return False


def _has_leadership_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when leadership framing names the cluster as a collective."""
    for tokens, index in _token_for_anchor(cluster, pre):
        if index >= 2:
            if (
                tokens[index - 2].text.lower() in GROUP_LEADERSHIP_NOUNS
                and tokens[index - 1].text.lower() == "of"
            ):
                return True
    return False


def _has_collective_action_context(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby verbs describe group-like collective action."""
    for tokens, index in _token_for_anchor(cluster, pre):
        right_window = tokens[index + 1:index + 4]
        if any(token.text.lower() in GROUP_COLLECTIVE_VERBS for token in right_window):
            return True
    return False


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
    reasons: list[str] = []
    score = 0.0

    if any(cluster.normalized_key.endswith(suffix) for suffix in FACTION_SUFFIXES):
        score = 0.75
        reasons.append("normalized key ends with a group-like suffix")

    if _has_membership_context(cluster, pre):
        score += 0.40
        reasons.append("appears in membership context")

    if _has_leadership_context(cluster, pre):
        score += 0.35
        reasons.append("appears in leadership framing")

    if _has_collective_action_context(cluster, pre):
        score += 0.35
        reasons.append("appears near a collective action verb")

    return ClassEvidence(
        category=LexiconCategory.GROUP,
        score=min(score, 1.0),
        reasons=reasons,
        vetoes=[],
    )
