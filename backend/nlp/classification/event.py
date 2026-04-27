"""
Event classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Event heuristics placeholder]
        B --> C[Event ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def score_event_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Return placeholder event evidence.

    Event-specific heuristics are introduced in a later phase. The scorer is
    still defined now so the arbitration surface can stabilize early.
    """
    del cluster, pre
    return ClassEvidence(
        category=LexiconCategory.EVENT,
        score=0.0,
        reasons=[],
        vetoes=[],
    )
