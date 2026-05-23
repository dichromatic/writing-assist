"""
Suppression rescue task builder - select rescue candidates and assemble task packets.

The deterministic extraction pipeline suppresses ~800+ entities. Most
suppressions are correct, but a subset are real entities that fail structural
heuristics. This module selects plausible rescue candidates and builds LLM
task packets for binary verification.

.. code-block:: mermaid

    flowchart TD
        A[ManuscriptReviewBundle] --> B[Filter suppressed entity records]
        B --> C{Rescue candidate?}
        C -->|Yes| D[Build evidence from raw text]
        C -->|No| E[Diagnostic: rejected]
        D --> F[LLMTaskPacket]
        E --> G[LLMTaskSelectionDiagnostic]
        F & G --> H[Packet + diagnostic lists]
"""

from __future__ import annotations

from backend.nlp.types import (
    DocumentEntityBucket,
    DocumentEntityRecord,
    DocumentStatus,
    DocumentType,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskSelectionDiagnostic,
    LexiconCategory,
    ManuscriptReviewBundle,
    SpanAnchor,
    SuppressReason,
    stable_hash_id,
)

_RESCUE_CONTEXT_RADIUS = 300
_RESCUE_EVIDENCE_LIMIT = 5

_RESCUABLE_REASONS = {
    SuppressReason.LOW_ENTITYHOOD,
    SuppressReason.COMPONENT_OVERLAP_NOISE,
    SuppressReason.GENERIC_LEXICAL_NOISE,
}

_SCHEMA_ID = "manuscript_suppression_rescue.v1"


def _rescue_candidate_selected(record: DocumentEntityRecord) -> tuple[bool, str]:
    """Return whether one suppressed record qualifies for LLM rescue triage.

    The gate narrows ~800+ suppressed records to ~30-50 candidates by
    requiring a rescuable suppression reason, sufficient occurrences, and
    either multi-scene presence or structural support.
    """
    if record.bucket != DocumentEntityBucket.SUPPRESSED:
        return False, "not_suppressed"
    if record.suppression_reason not in _RESCUABLE_REASONS:
        reason = record.suppression_reason.value if record.suppression_reason is not None else "none"
        return False, f"suppression_reason_{reason}_not_rescuable"
    if record.occurrence_count < 4:
        return False, "too_few_occurrences"

    has_structural_support = (
        record.has_title_support
        or record.attribution_count > 0
    )
    if record.scene_count < 2 and not has_structural_support:
        return False, "single_scene_no_structural_support"
    if (
        record.winning_category == LexiconCategory.UNRESOLVED
        and record.entityhood_score < 0.25
    ):
        return False, "unresolved_very_low_entityhood"
    return True, "rescue_candidate"


def _build_evidence_item(
    *,
    source_object_id: str,
    anchor: SpanAnchor,
    quote: str,
    context_before: str,
    context_after: str,
    suppression_reason: str,
    confidence_score: float | None,
) -> LLMTaskEvidenceItem:
    """Build one evidence item for a rescue task packet."""
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
        visibility_bucket="suppressed",
        suppression_reason=suppression_reason,
        confidence_score=confidence_score,
    )


def _build_rescue_evidence(
    record: DocumentEntityRecord,
    raw_text: str,
) -> list[LLMTaskEvidenceItem]:
    """Build evidence windows from anchors and raw document text.

    Slices context around each anchor position at assembly time so
    suppressed records do not need promotion-stage context windows.
    """
    items: list[LLMTaskEvidenceItem] = []
    sorted_anchors = sorted(record.anchors, key=lambda a: a.start_char)
    quote = record.surface_forms[0] if record.surface_forms else record.normalized_key
    suppression = record.suppression_reason.value if record.suppression_reason else ""

    for anchor in sorted_anchors[:_RESCUE_EVIDENCE_LIMIT]:
        before_start = max(0, anchor.start_char - _RESCUE_CONTEXT_RADIUS)
        after_end = min(len(raw_text), anchor.end_char + _RESCUE_CONTEXT_RADIUS)
        context_before = raw_text[before_start:anchor.start_char]
        context_after = raw_text[anchor.end_char:after_end]
        if not context_before.strip() and not context_after.strip():
            continue
        items.append(
            _build_evidence_item(
                source_object_id=record.normalized_key,
                anchor=anchor,
                quote=quote,
                context_before=context_before,
                context_after=context_after,
                suppression_reason=suppression,
                confidence_score=record.confidence_score,
            )
        )
    return items


def build_rescue_task_packets(
    bundle: ManuscriptReviewBundle,
    document_texts: dict[str, str],
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build suppression rescue LLM task packets from a manuscript bundle.

    Args:
        bundle: Manuscript review bundle containing entity records.
        document_texts: Map of document path to raw text content, used
            to build evidence windows around rescue candidate anchors.

    Returns:
        Task packets for rescue candidates and selection diagnostics
        for all suppressed records evaluated.
    """
    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []

    for record in bundle.entity_records:
        selected, reason = _rescue_candidate_selected(record)
        rescue_evidence: list[LLMTaskEvidenceItem] = []

        if selected:
            raw_text = document_texts.get(record.document_anchor.path, "")
            if not raw_text:
                selected = False
                reason = "missing_document_text"
            else:
                rescue_evidence = _build_rescue_evidence(record, raw_text)
                if not rescue_evidence:
                    selected = False
                    reason = "no_rescue_evidence_windows"

        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="suppressed_entity_record",
                source_object_id=record.normalized_key,
                document_path=record.document_anchor.path,
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                selected=selected,
                reason=reason,
                evidence_counts={
                    "occurrence_count": record.occurrence_count,
                    "scene_count": record.scene_count,
                    "anchor_count": len(record.anchors),
                },
            )
        )
        if not selected:
            continue

        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE.value,
                    record.normalized_key,
                    record.document_anchor.path,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                schema_id=_SCHEMA_ID,
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="suppressed_entity_record",
                source_object_id=record.normalized_key,
                source_document_paths=[record.document_anchor.path],
                document_type=DocumentType.MANUSCRIPT,
                document_status=DocumentStatus.PRIMARY_CANON,
                source_authority="manuscript_corpus",
                source_authority_weight=1.0,
                task_goal=(
                    "Determine whether this suppressed entity mention is a genuine "
                    "recurring entity that was incorrectly filtered by deterministic rules."
                ),
                task_constraints=[
                    "Use only the surrounding manuscript context provided in evidence.",
                    (
                        "A genuine entity is a named character, place, group, ship, "
                        "title-as-name, or concept that recurs meaningfully in the narrative."
                    ),
                    "Generic English words that happen to be capitalized are not entities.",
                    "If a title or rank consistently refers to one character, rescue it.",
                ],
                evidence_payload=rescue_evidence,
                selection_reason=reason,
                payload={
                    "normalized_key": record.normalized_key,
                    "surface_forms": list(record.surface_forms),
                    "occurrence_count": record.occurrence_count,
                    "scene_count": record.scene_count,
                    "suppression_reason": (
                        record.suppression_reason.value
                        if record.suppression_reason else ""
                    ),
                    "winning_category": record.winning_category.value,
                    "confidence_score": record.confidence_score,
                    "entityhood_score": record.entityhood_score,
                },
            )
        )

    return packets, diagnostics
