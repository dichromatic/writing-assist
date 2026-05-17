"""
Semantic review task surface - proposals and review prompts from prepared evidence.

.. code-block:: mermaid

    flowchart TD
        A[ReferenceCandidate list] --> B[ReferenceCluster list]
        C[CorpusEntity list] --> D[CharacterSemanticSummary list]
        B & D --> E[ReviewContext]
        E --> F[Build SemanticProposal list]
        E --> G[Build ReviewTask list]
"""

from __future__ import annotations

from backend.nlp.harvesting.shared import stable_hash_id
from backend.nlp.semantic_review.reference_extraction import extract_reference_candidates
from backend.nlp.semantic_review.review_helpers import (
    ReviewContext,
    _reference_evidence_note,
    _reference_resolution_context,
    _reference_review_prompt,
    build_character_summaries,
    build_conflict_records,
    build_reference_clusters,
    build_review_context,
)
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    ReferenceCluster,
    ReferenceCandidateType,
    ReviewTask,
    ReviewTaskKind,
    SemanticProposal,
    SemanticProposalConfidence,
    SemanticProposalSource,
)


def build_semantic_proposals(
    reference_clusters: list[ReferenceCluster],
    character_summaries: list[CharacterSemanticSummary] | None = None,
    review_context: ReviewContext | None = None,
) -> list[SemanticProposal]:
    """Build narrow structured proposals from strong semantic-review evidence.

    This pass only emits proposals when the current deterministic and
    corpus-backed evidence already points to one likely target. Ambiguous
    references remain review tasks only.
    """
    if (
        review_context is not None
        and character_summaries is not None
        and review_context.ownership_mode != "canonical_character_summaries"
    ):
        raise ValueError(
            "review_context built from raw reference clusters cannot be reused "
            "with canonical character summaries"
        )

    resolved_context = review_context or build_review_context(
        reference_clusters,
        character_summaries=character_summaries,
    )
    title_owner_scores = resolved_context.title_owner_scores
    relation_owner_scores = resolved_context.relation_owner_scores
    canonical_map = resolved_context.canonical_map

    proposals: list[SemanticProposal] = []
    for reference in reference_clusters:
        if reference.reference_type not in {
            ReferenceCandidateType.BARE_TITLE_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            continue

        context = _reference_resolution_context(
            reference,
            title_owner_scores,
            relation_owner_scores,
            canonical_map,
        )
        ranked_speaker_keys = context["ranked_speaker_keys"]
        ranked_entity_keys = context["ranked_entity_keys"]
        non_speaker_entity_keys = context["non_speaker_entity_keys"]
        dominant_owner_key = context["dominant_owner_key"]

        proposed_target_key: str | None = None
        source: SemanticProposalSource | None = None
        rationale: str | None = None
        if reference.address_like_count and ranked_speaker_keys and len(non_speaker_entity_keys) == 1:
            proposed_target_key = non_speaker_entity_keys[0]
            source = SemanticProposalSource.ADDRESS_LOCAL_CONTEXT
            rationale = (
                f"address-like reference spoken by {', '.join(ranked_speaker_keys)} "
                f"has one non-speaker local target"
            )
        elif reference.address_like_count and ranked_speaker_keys and dominant_owner_key is not None:
            proposed_target_key = dominant_owner_key
            source = SemanticProposalSource.DOMINANT_OWNER
            rationale = (
                f"address-like reference spoken by {', '.join(ranked_speaker_keys)} "
                f"has one dominant corpus owner"
            )
        elif len(ranked_entity_keys) == 1:
            proposed_target_key = ranked_entity_keys[0]
            source = SemanticProposalSource.LOCAL_CONTEXT
            rationale = "local deterministic evidence points to one canonical target"

        if proposed_target_key is None or source is None or rationale is None:
            continue

        proposals.append(SemanticProposal(
            proposal_id=stable_hash_id(
                reference.document_anchor.path,
                reference.reference_type.value,
                reference.normalized,
                proposed_target_key,
                source.value,
            ),
            reference_type=reference.reference_type,
            subject_key=reference.normalized,
            document_anchor=reference.document_anchor,
            proposed_target_key=proposed_target_key,
            source=source,
            confidence=SemanticProposalConfidence.LIKELY,
            supporting_anchors=reference.anchors,
            rationale=rationale,
        ))

    return sorted(
        proposals,
        key=lambda proposal: (
            proposal.reference_type.value,
            proposal.document_anchor.path,
            proposal.subject_key,
            proposal.proposed_target_key,
        ),
    )


def build_review_tasks(
    reference_clusters: list[ReferenceCluster],
    conflicts: list[ConflictRecord],
    character_summaries: list[CharacterSemanticSummary] | None = None,
    review_context: ReviewContext | None = None,
) -> list[ReviewTask]:
    """Build lightweight semantic review prompts from structured evidence.

    Args:
        reference_clusters: Grouped deferred title and role references.
        conflicts: Typed cross-category conflict records.

    Returns:
        Stable review tasks for later human or LLM review.
    """
    deduped_tasks: dict[tuple[str, str, str, tuple[str, ...]], ReviewTask] = {}
    if (
        review_context is not None
        and character_summaries is not None
        and review_context.ownership_mode != "canonical_character_summaries"
    ):
        raise ValueError(
            "review_context built from raw reference clusters cannot be reused "
            "with canonical character summaries"
        )

    resolved_context = review_context or build_review_context(
        reference_clusters,
        character_summaries=character_summaries,
    )
    title_owner_scores = resolved_context.title_owner_scores
    relation_owner_scores = resolved_context.relation_owner_scores
    canonical_map = resolved_context.canonical_map

    for reference in reference_clusters:
        if reference.reference_type not in {
            ReferenceCandidateType.BARE_TITLE_ROLE,
            ReferenceCandidateType.BARE_RELATION_ROLE,
        }:
            continue
        context = _reference_resolution_context(
            reference,
            title_owner_scores,
            relation_owner_scores,
            canonical_map,
        )
        ranked_entity_keys = context["ranked_entity_keys"]
        ranked_speaker_keys = context["ranked_speaker_keys"]
        non_speaker_entity_keys = context["non_speaker_entity_keys"]
        corpus_owner_keys = context["corpus_owner_keys"]
        dominant_owner_key = context["dominant_owner_key"]
        if reference.reference_type == ReferenceCandidateType.BARE_TITLE_ROLE:
            kind = ReviewTaskKind.TITLE_ROLE_ATTACHMENT
            label = "title"
        else:
            kind = ReviewTaskKind.RELATION_ROLE_ATTACHMENT
            label = "relation"
        evidence_note = _reference_evidence_note(
            reference,
            ranked_entity_keys,
            ranked_speaker_keys,
            corpus_owner_keys,
            dominant_owner_key,
        )
        prompt = _reference_review_prompt(
            reference,
            label,
            ranked_entity_keys,
            ranked_speaker_keys,
            non_speaker_entity_keys,
            corpus_owner_keys,
            dominant_owner_key,
        )
        task = ReviewTask(
            task_id=stable_hash_id(
                reference.document_anchor.path,
                reference.reference_type.value,
                reference.normalized,
            ),
            kind=kind,
            subject_key=reference.normalized,
            prompt=prompt,
            supporting_anchor_paths=[reference.document_anchor.path],
            ranked_candidate_keys=ranked_entity_keys,
            ranked_speaker_keys=ranked_speaker_keys,
            corpus_owner_keys=corpus_owner_keys,
            evidence_note=evidence_note,
        )
        deduped_tasks[(
            task.kind.value,
            task.subject_key,
            task.prompt,
            tuple(task.supporting_anchor_paths),
        )] = task

    for conflict in conflicts:
        task = ReviewTask(
            task_id=stable_hash_id(
                conflict.canonical_key,
                ",".join(category.value for category in conflict.conflicting_categories),
            ),
            kind=ReviewTaskKind.CATEGORY_CONFLICT,
            subject_key=conflict.canonical_key,
            prompt=(
                f"Resolve the category conflict for '{conflict.canonical_key}': "
                f"{', '.join(category.value for category in conflict.conflicting_categories)}"
            ),
            supporting_anchor_paths=conflict.supporting_document_paths,
            evidence_note=conflict.reason,
        )
        deduped_tasks[(
            task.kind.value,
            task.subject_key,
            task.prompt,
            tuple(task.supporting_anchor_paths),
        )] = task

    return sorted(
        deduped_tasks.values(),
        key=lambda task: (task.kind.value, task.subject_key, task.prompt),
    )
