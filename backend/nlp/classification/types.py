"""
Classification contracts for top-level entity categories.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster + document context] --> B[Per-category evidence scorer]
        B --> C[ClassEvidence]
        C --> D[Arbitration]
        D --> E[ClassificationDecision]
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.nlp.types import LexiconCategory


@dataclass(frozen=True)
class ClassEvidence:
    """Evidence and score for one top-level classification.

    Args:
        category: The category this evidence describes.
        score: Deterministic confidence score for this category in [0.0, 1.0].
        reasons: Human-readable reasons that raised the score.
        vetoes: Human-readable reasons that prevented or reduced confidence.
    """

    category: LexiconCategory
    score: float
    reasons: list[str]
    vetoes: list[str]


@dataclass(frozen=True)
class EntityhoodDecision:
    """Whether a cluster is strong enough to survive as an entity candidate.

    Args:
        score: Deterministic entityhood score in [0.0, 1.0].
        accepted: True when the cluster is plausible enough to survive as an
            entity candidate even if its top-level class remains unresolved.
        reasons: Human-readable reasons that raised entityhood confidence.
        weaknesses: Human-readable reasons that kept entityhood weak.
    """

    score: float
    accepted: bool
    reasons: list[str]
    weaknesses: list[str]


@dataclass(frozen=True)
class ClassificationDecision:
    """Final top-level classification choice for one cluster.

    Args:
        winning_category: Highest-scoring category after arbitration.
        winning_score: Score for the winning category.
        runner_up_category: Next-best category, if any.
        runner_up_score: Score for the runner-up category.
        evidence_by_category: Raw evidence for every category scorer run.
        entityhood: Whether the cluster is plausible enough to survive as an
            entity candidate independent of top-level class resolution.
        resolved: True when the winning category cleared the resolution
            threshold and beat the runner-up by a stable margin.
    """

    winning_category: LexiconCategory
    winning_score: float
    runner_up_category: LexiconCategory | None
    runner_up_score: float
    evidence_by_category: dict[LexiconCategory, ClassEvidence]
    entityhood: EntityhoodDecision
    resolved: bool
