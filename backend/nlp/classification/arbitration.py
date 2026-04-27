"""
Top-level classification arbitration across type-specific evidence modules.

.. code-block:: mermaid

    flowchart TD
        A[MentionCluster + document context] --> B[Character scorer]
        A --> C[Group scorer]
        A --> D[Place scorer]
        A --> E[Object scorer]
        A --> F[Event scorer]
        A --> G[Concept scorer]
        B & C & D & E & F & G --> H[Arbitrate scores and margins]
        H --> I[ClassificationDecision]
"""

from __future__ import annotations

from backend.nlp.classification.character import score_character_evidence
from backend.nlp.classification.concept import score_concept_evidence
from backend.nlp.classification.entityhood import assess_entityhood
from backend.nlp.classification.event import score_event_evidence
from backend.nlp.classification.group import score_group_evidence
from backend.nlp.classification.object import score_object_evidence
from backend.nlp.classification.place import score_place_evidence
from backend.nlp.classification.types import (
    ClassEvidence,
    ClassificationDecision,
    EntityhoodDecision,
)
from backend.nlp.types import LexiconCategory, MentionCluster, PreprocessedDocument

_RESOLUTION_THRESHOLD = 0.60
_RESOLUTION_MARGIN = 0.10


def _build_evidence_map(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
    attributed_speakers: frozenset[str],
) -> dict[LexiconCategory, ClassEvidence]:
    """Run every top-level evidence scorer for one cluster."""
    return {
        LexiconCategory.CHARACTER: score_character_evidence(cluster, pre, attributed_speakers),
        LexiconCategory.GROUP: score_group_evidence(cluster, pre),
        LexiconCategory.PLACE: score_place_evidence(cluster, pre, attributed_speakers),
        LexiconCategory.OBJECT: score_object_evidence(cluster, pre),
        LexiconCategory.EVENT: score_event_evidence(cluster, pre),
        LexiconCategory.CONCEPT: score_concept_evidence(cluster, pre),
    }


def _resolve_decision(
    evidence_by_category: dict[LexiconCategory, ClassEvidence],
) -> ClassificationDecision:
    """Choose the winning category from per-class evidence."""
    ranked = sorted(
        evidence_by_category.values(),
        key=lambda evidence: (evidence.score, evidence.category.value),
        reverse=True,
    )

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    runner_up_score = runner_up.score if runner_up is not None else 0.0
    resolved = (
        winner.score >= _RESOLUTION_THRESHOLD
        and winner.score - runner_up_score >= _RESOLUTION_MARGIN
    )

    return ClassificationDecision(
        winning_category=winner.category if resolved else LexiconCategory.UNRESOLVED,
        winning_score=winner.score,
        runner_up_category=runner_up.category if runner_up is not None else None,
        runner_up_score=runner_up_score,
        evidence_by_category=evidence_by_category,
        entityhood=EntityhoodDecision(
            score=0.0,
            accepted=False,
            reasons=[],
            weaknesses=[],
        ),
        resolved=resolved,
    )


def classify_cluster(
    cluster: MentionCluster,
    pre: PreprocessedDocument | None,
    attribution_records: list,
) -> ClassificationDecision:
    """Classify one cluster using shared document context.

    Args:
        cluster: Cluster being classified.
        pre: Preprocessed document context.
        attribution_records: Speaker attribution records. Records only need a
            ``speaker_key`` attribute for this phase.

    Returns:
        A top-level classification decision for the cluster.
    """
    attributed_speakers = frozenset(record.speaker_key for record in attribution_records)
    evidence_by_category = _build_evidence_map(cluster, pre, attributed_speakers)
    decision = _resolve_decision(evidence_by_category)
    return ClassificationDecision(
        winning_category=decision.winning_category,
        winning_score=decision.winning_score,
        runner_up_category=decision.runner_up_category,
        runner_up_score=decision.runner_up_score,
        evidence_by_category=decision.evidence_by_category,
        entityhood=assess_entityhood(cluster, evidence_by_category),
        resolved=decision.resolved,
    )


def classify_clusters(
    clusters: list[MentionCluster],
    pre: PreprocessedDocument | None,
    attribution_records: list,
) -> dict[str, ClassificationDecision]:
    """Classify every cluster in the document.

    Args:
        clusters: Clusters to classify.
        pre: Preprocessed document context.
        attribution_records: Speaker attribution records.

    Returns:
        Mapping from normalized cluster key to classification decision.
    """
    return {
        cluster.normalized_key: classify_cluster(cluster, pre, attribution_records)
        for cluster in clusters
    }
