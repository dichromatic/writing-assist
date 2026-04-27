"""
Object classification evidence.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster] --> B[Object heuristics placeholder]
        B --> C[Object ClassEvidence]
"""

from __future__ import annotations

from backend.nlp.classification.types import ClassEvidence
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument


def score_object_evidence(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
) -> ClassEvidence:
    """Return placeholder object evidence.

    Object-specific heuristics are introduced in a later phase. The scorer is
    still defined now so the arbitration surface can stabilize early.
    """
    del cluster, pre
    return ClassEvidence(
        category=LexiconCategory.OBJECT,
        score=0.0,
        reasons=[],
        vetoes=[],
    )
