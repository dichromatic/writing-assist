"""
Concept classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Linked definitions]
        A --> C[Definition syntax]
        A --> D[Abstract descriptor noun]
        B & C & D --> E[Concept ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    CONCEPT_DESCRIPTOR_NOUNS,
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


def _has_definition_syntax(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when local context defines the cluster as a named concept."""
    for tokens, index in _token_for_anchor(cluster, pre):
        right_window = [token.text.lower() for token in tokens[index + 1:index + 6]]
        if not right_window:
            continue

        if len(right_window) >= 2 and right_window[0] == "refers" and right_window[1] == "to":
            return True

        if (
            right_window[0] in {"is", "was", "means", "meant", "describes", "described", "denotes", "denoted"}
            and any(word in CONCEPT_DESCRIPTOR_NOUNS for word in right_window[1:])
        ):
            return True

        if (
            right_window[0] in {"describe", "describes", "described"}
            and any(word in {"term", "concept"} for word in right_window[1:])
        ):
            return True

    return False


def _has_concept_descriptor(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby nouns frame the cluster as an abstract system or term."""
    for tokens, index in _token_for_anchor(cluster, pre):
        right_window = [token.text.lower() for token in tokens[index + 1:index + 8]]
        if any(word in CONCEPT_DESCRIPTOR_NOUNS for word in right_window):
            return True

        left_window = [token.text.lower() for token in tokens[max(0, index - 3):index]]
        if any(word in {"term", "concept"} for word in left_window):
            return True

    return False


def score_concept_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Score how strongly a cluster behaves like an abstract named concept.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context.

    Returns:
        Concept evidence for the cluster.
    """
    score = 0.0
    reasons: list[str] = []
    vetoes: list[str] = []

    if cluster.linked_definitions:
        score += 0.80
        reasons.append("is referenced by definition-style notes")

    if _has_definition_syntax(cluster, pre):
        score += 0.40
        reasons.append("appears in explicit definition syntax")

    if _has_concept_descriptor(cluster, pre):
        score += 0.30
        reasons.append("appears near an abstract descriptor noun")

    return ClassEvidence(
        category=LexiconCategory.CONCEPT,
        score=min(score, 1.0),
        reasons=reasons,
        vetoes=vetoes,
    )
