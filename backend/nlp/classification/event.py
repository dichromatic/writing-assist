"""
Event classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Event noun cue]
        A --> C[Temporal framing]
        A --> D[Occurrence verb cue]
        B & C & D --> E[Event ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    EVENT_INSTANCE_MARKERS,
    EVENT_NOUNS,
    EVENT_OCCURRENCE_VERBS,
    EVENT_TEMPORAL_PREPOSITIONS,
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


def _has_temporal_framing(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when the cluster appears in explicit event-time framing."""
    for tokens, index in _token_for_anchor(cluster, pre):
        if index >= 1 and tokens[index - 1].text.lower() in EVENT_TEMPORAL_PREPOSITIONS:
            return True
        if (
            index >= 2
            and tokens[index - 2].text.lower() in EVENT_TEMPORAL_PREPOSITIONS
            and tokens[index - 1].text.lower() in {"the", "a", "an", "this", "that"}
        ):
            return True
    return False


def _has_instance_marker(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby modifiers frame the cluster as a recurring event."""
    for tokens, index in _token_for_anchor(cluster, pre):
        if index >= 1 and tokens[index - 1].text.lower() in EVENT_INSTANCE_MARKERS:
            return True
        if (
            index >= 2
            and tokens[index - 2].text.lower() in EVENT_INSTANCE_MARKERS
            and tokens[index - 1].text.lower() in {"the", "a", "an"}
        ):
            return True
    return False


def _has_occurrence_verb(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby verbs describe the event as happening or being held."""
    for tokens, index in _token_for_anchor(cluster, pre):
        right_window = tokens[index + 1:index + 4]
        left_window = tokens[max(0, index - 3):index]
        if any(token.text.lower() in EVENT_OCCURRENCE_VERBS for token in right_window):
            return True
        if any(token.text.lower() in EVENT_OCCURRENCE_VERBS for token in left_window):
            return True
    return False


def score_event_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Score how strongly a cluster behaves like a named event.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context.

    Returns:
        Event evidence for the cluster.
    """
    score = 0.0
    reasons: list[str] = []
    vetoes: list[str] = []

    if cluster.normalized_key in EVENT_NOUNS:
        score += 0.35
        reasons.append("normalized key is an event-like noun")

    if _has_temporal_framing(cluster, pre):
        score += 0.35
        reasons.append("appears in explicit temporal event framing")

    if _has_occurrence_verb(cluster, pre):
        score += 0.35
        reasons.append("appears near an occurrence or observance verb")

    if _has_instance_marker(cluster, pre):
        score += 0.20
        reasons.append("appears with a recurring-event marker")

    return ClassEvidence(
        category=LexiconCategory.EVENT,
        score=min(score, 1.0),
        reasons=reasons,
        vetoes=vetoes,
    )
