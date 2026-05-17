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

from backend.nlp.classification.compound_shapes import compound_head
from backend.nlp.classification.scoring_builder import ScoringBuilder
from backend.nlp.classification.token_context import has_anchor_token_pattern
from backend.nlp.classification.types import ClassEvidence
from backend.nlp.harvesting.shared import (
    EVENT_INSTANCE_MARKERS,
    EVENT_NOUNS,
    EVENT_OCCURRENCE_VERBS,
    EVENT_TEMPORAL_PREPOSITIONS,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument, Token


def _has_temporal_framing(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when the cluster appears in explicit event-time framing."""
    def _matches(tokens: list[Token], index: int) -> bool:
        if index >= 1 and tokens[index - 1].text.lower() in EVENT_TEMPORAL_PREPOSITIONS:
            return True
        if (
            index >= 2
            and tokens[index - 2].text.lower() in EVENT_TEMPORAL_PREPOSITIONS
            and tokens[index - 1].text.lower() in {"the", "a", "an", "this", "that"}
        ):
            return True
        return False

    return has_anchor_token_pattern(cluster, pre, _matches)


def _has_instance_marker(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby modifiers frame the cluster as a recurring event."""
    def _matches(tokens: list[Token], index: int) -> bool:
        if index >= 1 and tokens[index - 1].text.lower() in EVENT_INSTANCE_MARKERS:
            return True
        if (
            index >= 2
            and tokens[index - 2].text.lower() in EVENT_INSTANCE_MARKERS
            and tokens[index - 1].text.lower() in {"the", "a", "an"}
        ):
            return True
        return False

    return has_anchor_token_pattern(cluster, pre, _matches)


def _has_occurrence_verb(cluster: MentionCluster, pre: PreprocessedDocument | None) -> bool:
    """Return True when nearby verbs describe the event as happening or being held."""
    def _matches(tokens: list[Token], index: int) -> bool:
        right_window = tokens[index + 1:index + 4]
        left_window = tokens[max(0, index - 3):index]
        if any(token.text.lower() in EVENT_OCCURRENCE_VERBS for token in right_window):
            return True
        return any(token.text.lower() in EVENT_OCCURRENCE_VERBS for token in left_window)

    return has_anchor_token_pattern(cluster, pre, _matches)


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
    builder = ScoringBuilder(LexiconCategory.EVENT)

    head = compound_head(cluster)

    if cluster.normalized_key in EVENT_NOUNS:
        builder.add(0.35, "normalized key is an event-like noun")
    elif head in EVENT_NOUNS:
        builder.add(0.65, "compound head is an event-like noun")

    if _has_temporal_framing(cluster, pre):
        builder.add(0.35, "appears in explicit temporal event framing")

    if _has_occurrence_verb(cluster, pre):
        builder.add(0.35, "appears near an occurrence or observance verb")

    if _has_instance_marker(cluster, pre):
        builder.add(0.20, "appears with a recurring-event marker")

    return builder.build()
