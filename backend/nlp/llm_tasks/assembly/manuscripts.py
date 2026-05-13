"""
Manuscript LLM task builder - manuscript review bundle to shared task packets.

.. code-block:: mermaid

    flowchart TD
        A[ManuscriptReviewBundle] --> B[Evaluate entity/reference/conflict selectors]
        B --> C[Build LLMTaskPacket list]
        B --> D[Build LLMTaskSelectionDiagnostic list]
"""

from __future__ import annotations

from collections import defaultdict

from backend.nlp.document_metadata import document_status_authority_weight
from backend.nlp.llm_tasks.assembly.schemas import schema_id_for
from backend.nlp.types import (
    ConflictRecord,
    CorpusEntity,
    LLMTaskEvidenceItem,
    LLMTaskPacket,
    LLMTaskFamily,
    LLMTaskSelectionDiagnostic,
    ManuscriptReviewBundle,
    ReferenceCluster,
    ReferenceCandidateType,
    ReviewTaskKind,
    SpanAnchor,
    stable_hash_id,
)

_ENTITY_EVIDENCE_LIMIT = 10
_REFERENCE_EVIDENCE_LIMIT = 12
_CONFLICT_EVIDENCE_LIMIT = 8


def _evidence_item(
    *,
    source_object_id: str,
    anchor: SpanAnchor,
    quote: str,
    visibility_bucket: str,
    context_before: str,
    context_after: str,
    suppression_reason: str = "",
    confidence_score: float | None = None,
) -> LLMTaskEvidenceItem:
    """Build one bounded evidence item for a manuscript task packet."""
    return LLMTaskEvidenceItem(
        evidence_id=stable_hash_id(
            "llm_task_evidence",
            source_object_id,
            anchor.path,
            str(anchor.start_char),
            str(anchor.end_char),
            quote,
        ),
        document_path=anchor.path,
        source_anchor=anchor,
        quote=quote,
        context_before=context_before,
        context_after=context_after,
        source_object_id=source_object_id,
        visibility_bucket=visibility_bucket,
        suppression_reason=suppression_reason,
        confidence_score=confidence_score,
    )


def _entity_selected(entity: CorpusEntity) -> tuple[bool, str]:
    """Return whether a corpus entity should receive an entity-triage task."""
    if entity.canonical_key in {"everyone"}:
        return False, "generic_pronoun_noise"
    if entity.review_required:
        return True, "review_required_conflict"
    if len(entity.supporting_document_paths) > 1:
        return True, "multi_document_support"
    if entity.absorbed_surface_forms:
        return True, "absorbed_surface_forms"
    if len(entity.source_keys) > 1:
        return True, "multiple_source_keys"
    return False, "thin_unresolved_entity"


def _reference_normalized_keys(clusters: list[ReferenceCluster]) -> set[str]:
    """Return normalized reference keys that should route to attachment lane."""
    routed_types = {
        ReferenceCandidateType.BOUND_TITLE_ROLE,
        ReferenceCandidateType.BARE_TITLE_ROLE,
        ReferenceCandidateType.BOUND_RELATION_ROLE,
        ReferenceCandidateType.BARE_RELATION_ROLE,
    }
    return {
        cluster.normalized
        for cluster in clusters
        if cluster.reference_type in routed_types
    }


def _entity_evidence(entity: CorpusEntity) -> list[LLMTaskEvidenceItem]:
    """Build bounded evidence for one corpus entity profile task."""
    items: list[LLMTaskEvidenceItem] = []
    dropped_no_context = 0
    for record in entity.member_records[:_ENTITY_EVIDENCE_LIMIT]:
        if not record.anchors or not record.evidence_windows:
            continue
        anchor = record.anchors[0]
        window = next(
            (
                item
                for item in record.evidence_windows
                if (
                    item.anchor.path == anchor.path
                    and item.anchor.span_ordinal == anchor.span_ordinal
                    and item.anchor.start_char == anchor.start_char
                    and item.anchor.end_char == anchor.end_char
                )
            ),
            None,
        )
        if window is None:
            dropped_no_context += 1
            continue
        if not window.context_before.strip() and not window.context_after.strip():
            dropped_no_context += 1
            continue
        quote = record.surface_forms[0] if record.surface_forms else record.normalized_key
        items.append(
            _evidence_item(
                source_object_id=entity.canonical_key,
                anchor=anchor,
                quote=quote,
                visibility_bucket=record.bucket.value,
                context_before=window.context_before,
                context_after=window.context_after,
                suppression_reason=(
                    record.suppression_reason.value
                    if record.suppression_reason is not None
                    else ""
                ),
                confidence_score=record.confidence_score,
            )
        )
    if dropped_no_context and not items:
        # Keep one weak fallback item when every window was dropped so we do
        # not erase the task entirely; later quality gates can decide reviewability.
        record = entity.member_records[0]
        if record.anchors:
            quote = record.surface_forms[0] if record.surface_forms else record.normalized_key
            items.append(
                _evidence_item(
                    source_object_id=entity.canonical_key,
                    anchor=record.anchors[0],
                    quote=quote,
                    visibility_bucket=record.bucket.value,
                    context_before="",
                    context_after="",
                    suppression_reason=(
                        record.suppression_reason.value
                        if record.suppression_reason is not None
                        else ""
                    ),
                    confidence_score=record.confidence_score,
                )
            )
    return items


def _reference_selected(cluster: ReferenceCluster) -> tuple[bool, str]:
    """Return whether one reference cluster should receive an attachment task."""
    if cluster.candidate_entity_scores or cluster.speaker_entity_scores:
        return True, "ranked_candidate_keys_present"
    if cluster.occurrence_count > 1:
        return True, "recurring_reference_cluster"
    return False, "no_attachment_signal"


def _reference_evidence(cluster: ReferenceCluster) -> list[LLMTaskEvidenceItem]:
    """Build bounded evidence for one reference-attachment task."""
    items: list[LLMTaskEvidenceItem] = []
    for anchor in cluster.anchors[:_REFERENCE_EVIDENCE_LIMIT]:
        items.append(
            _evidence_item(
                source_object_id=f"{cluster.reference_type.value}:{cluster.normalized}",
                anchor=anchor,
                quote=cluster.surface_forms[0] if cluster.surface_forms else cluster.normalized,
                visibility_bucket=cluster.reference_type.value,
                context_before="",
                context_after="",
            )
        )
    return items


def _conflict_evidence(
    conflict: ConflictRecord,
    entities_by_key: dict[str, CorpusEntity],
) -> list[LLMTaskEvidenceItem]:
    """Build bounded evidence for one category-resolution task."""
    entity = entities_by_key.get(conflict.canonical_key)
    if entity is None:
        return []
    items: list[LLMTaskEvidenceItem] = []
    for record in entity.member_records[:_CONFLICT_EVIDENCE_LIMIT]:
        if not record.anchors:
            continue
        items.append(
            _evidence_item(
                source_object_id=conflict.canonical_key,
                anchor=record.anchors[0],
                quote=record.surface_forms[0] if record.surface_forms else record.normalized_key,
                visibility_bucket="conflict_support",
                context_before="",
                context_after="",
                confidence_score=record.confidence_score,
            )
        )
    return items


def _review_task_index(bundle: ManuscriptReviewBundle) -> dict[str, list[str]]:
    """Index review prompts by subject key for packet payload context."""
    indexed: dict[str, list[str]] = defaultdict(list)
    for task in bundle.review_tasks:
        if task.kind not in {
            ReviewTaskKind.TITLE_ROLE_ATTACHMENT,
            ReviewTaskKind.RELATION_ROLE_ATTACHMENT,
        }:
            continue
        indexed[task.subject_key].append(task.prompt)
    return indexed


def _status_for_entity(bundle: ManuscriptReviewBundle) -> str:
    """Return conservative manuscript status metadata for task packets."""
    _ = bundle
    return "primary_canon"


def _status_for_reference(bundle: ManuscriptReviewBundle) -> str:
    """Return conservative manuscript status metadata for task packets."""
    _ = bundle
    return "primary_canon"


def _status_for_conflict(bundle: ManuscriptReviewBundle) -> str:
    """Return conservative manuscript status metadata for task packets."""
    _ = bundle
    return "primary_canon"

from backend.nlp.types import (
    DocumentStatus,
    DocumentType,
)


def build_manuscript_task_packets(
    bundle: ManuscriptReviewBundle,
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build manuscript LLM task packets with explicit selection diagnostics."""
    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []
    entities_by_key = {entity.canonical_key: entity for entity in bundle.canonical_entities}
    review_prompts_by_reference = _review_task_index(bundle)
    reference_keys = _reference_normalized_keys(bundle.reference_clusters)

    for entity in bundle.canonical_entities:
        selected, reason = _entity_selected(entity)
        if selected and entity.canonical_key in reference_keys and not entity.review_required:
            selected, reason = False, "routed_to_reference_attachment"
        entity_evidence = _entity_evidence(entity) if selected else []
        context_kept = sum(
            1
            for item in entity_evidence
            if item.context_before.strip() or item.context_after.strip()
        )
        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="corpus_entity",
                source_object_id=entity.canonical_key,
                document_path=entity.supporting_document_paths[0] if entity.supporting_document_paths else "",
                task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
                selected=selected,
                reason=reason,
                evidence_counts={
                    "member_records": len(entity.member_records),
                    "supporting_documents": len(entity.supporting_document_paths),
                    "absorbed_surface_forms": len(entity.absorbed_surface_forms),
                    "context_kept": context_kept,
                    "context_dropped": max(0, len(entity.member_records[:_ENTITY_EVIDENCE_LIMIT]) - context_kept),
                },
            )
        )
        if not selected:
            continue
        status = DocumentStatus(_status_for_entity(bundle))
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE.value,
                    entity.canonical_key,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
                schema_id=schema_id_for(LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE),
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="corpus_entity",
                source_object_id=entity.canonical_key,
                source_document_paths=list(entity.supporting_document_paths),
                document_type=DocumentType.MANUSCRIPT,
                document_status=status,
                source_authority="manuscript_corpus",
                source_authority_weight=document_status_authority_weight(status),
                task_goal="Triage deterministic entity categorization using evidence-backed review signals.",
                task_constraints=[
                    "Use only evidence from provided anchors.",
                    "Keep unresolved or conflicting signals explicit.",
                    "Do not write narrative profiles or summaries in this pass.",
                ],
                evidence_payload=entity_evidence,
                selection_reason=reason,
                payload={
                    "canonical_key": entity.canonical_key,
                    "source_keys": list(entity.source_keys),
                    "dominant_category": entity.dominant_category.value,
                    "review_required": entity.review_required,
                    "conflicting_categories": [item.value for item in entity.conflicting_categories],
                    "reasons": list(entity.reasons),
                },
            )
        )

    for cluster in bundle.reference_clusters:
        source_id = f"{cluster.reference_type.value}:{cluster.normalized}:{cluster.document_anchor.path}"
        selected, reason = _reference_selected(cluster)
        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="reference_cluster",
                source_object_id=source_id,
                document_path=cluster.document_anchor.path,
                task_family=LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT,
                selected=selected,
                reason=reason,
                evidence_counts={
                    "occurrence_count": cluster.occurrence_count,
                    "candidate_entity_scores": len(cluster.candidate_entity_scores),
                    "speaker_entity_scores": len(cluster.speaker_entity_scores),
                },
            )
        )
        if not selected:
            continue
        status = DocumentStatus(_status_for_reference(bundle))
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT.value,
                    source_id,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT,
                schema_id=schema_id_for(LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT),
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="reference_cluster",
                source_object_id=source_id,
                source_document_paths=[cluster.document_anchor.path],
                document_type=DocumentType.MANUSCRIPT,
                document_status=status,
                source_authority="manuscript_corpus",
                source_authority_weight=document_status_authority_weight(status),
                task_goal="Propose reference attachments and explicit ambiguity handling.",
                task_constraints=[
                    "Rank plausible attachments from provided candidates.",
                    "Reject unsupported attachment candidates.",
                    "Escalate unresolved ambiguity to open review questions.",
                ],
                evidence_payload=_reference_evidence(cluster),
                selection_reason=reason,
                payload={
                    "reference_type": cluster.reference_type.value,
                    "normalized": cluster.normalized,
                    "surface_forms": list(cluster.surface_forms),
                    "candidate_entity_scores": dict(cluster.candidate_entity_scores),
                    "speaker_entity_scores": dict(cluster.speaker_entity_scores),
                    "review_prompts": review_prompts_by_reference.get(cluster.normalized, []),
                },
            )
        )

    for conflict in bundle.conflict_records:
        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="conflict_record",
                source_object_id=conflict.canonical_key,
                document_path=conflict.supporting_document_paths[0] if conflict.supporting_document_paths else "",
                task_family=LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION,
                selected=True,
                reason="review_required_conflict",
                evidence_counts={
                    "conflicting_categories": len(conflict.conflicting_categories),
                    "supporting_documents": len(conflict.supporting_document_paths),
                },
            )
        )
        status = DocumentStatus(_status_for_conflict(bundle))
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION.value,
                    conflict.canonical_key,
                    conflict.reason,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION,
                schema_id=schema_id_for(LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION),
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="conflict_record",
                source_object_id=conflict.canonical_key,
                source_document_paths=list(conflict.supporting_document_paths),
                document_type=DocumentType.MANUSCRIPT,
                document_status=status,
                source_authority="manuscript_corpus",
                source_authority_weight=document_status_authority_weight(status),
                task_goal="Propose category resolution while preserving conflict uncertainty.",
                task_constraints=[
                    "Do not force single-category certainty when conflict evidence remains.",
                    "Ground every proposal in supplied conflict evidence.",
                ],
                evidence_payload=_conflict_evidence(conflict, entities_by_key),
                selection_reason="review_required_conflict",
                payload={
                    "canonical_key": conflict.canonical_key,
                    "source": conflict.source.value,
                    "conflicting_categories": [item.value for item in conflict.conflicting_categories],
                    "reason": conflict.reason,
                },
            )
        )

    return packets, diagnostics
