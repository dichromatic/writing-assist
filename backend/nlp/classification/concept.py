"""
Concept classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Concept heuristics placeholder]
        B --> C[Concept ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def score_concept_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Return placeholder concept evidence.

    Concept-specific heuristics are introduced in a later phase. The scorer is
    still defined now so the arbitration surface can stabilize early.
    """
    del cluster, pre
    return ClassEvidence(
        category=LexiconCategory.CONCEPT,
        score=0.0,
        reasons=[],
        vetoes=[],
    )
