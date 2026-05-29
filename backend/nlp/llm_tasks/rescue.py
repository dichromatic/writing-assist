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

from dataclasses import dataclass

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


def _rescue_candidate_selected(
    record: DocumentEntityRecord,
    absorbed_keys: frozenset[str],
) -> tuple[bool, str]:
    """Return whether one suppressed record qualifies for LLM rescue triage.

    The gate narrows ~800+ suppressed records to ~30-50 candidates by
    requiring a rescuable suppression reason, sufficient occurrences, and
    no prior absorption into a compound entity.
    """
    if record.current_state.bucket != DocumentEntityBucket.SUPPRESSED:
        return False, "not_suppressed"
    if record.identity.normalized_key in absorbed_keys:
        return False, "already_absorbed_into_compound"
    if record.promotion_trace.suppression_reason not in _RESCUABLE_REASONS:
        reason = record.promotion_trace.suppression_reason.value if record.promotion_trace.suppression_reason is not None else "none"
        return False, f"suppression_reason_{reason}_not_rescuable"
    # Generic lexical noise needs stronger signal to justify an LLM call.
    # At occ=3, words like "let", "come", "see" dominate; real entities
    # suppressed as generic noise (pioneer, explorer) have higher counts.
    min_occurrences = 4 if record.promotion_trace.suppression_reason == SuppressReason.GENERIC_LEXICAL_NOISE else 2
    if record.source_evidence.occurrence_count < min_occurrences:
        return False, "too_few_occurrences"

    # Single-scene entities are allowed through. Characters in flashback
    # chapters or ships mentioned only in one scene still deserve LLM triage.
    if (
        record.current_state.winning_category == LexiconCategory.UNRESOLVED
        and record.classification_trace.entityhood.score < 0.25
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
    sorted_anchors = sorted(record.source_evidence.anchors, key=lambda a: a.start_char)
    quote = record.identity.surface_forms[0] if record.identity.surface_forms else record.identity.normalized_key
    suppression = record.promotion_trace.suppression_reason.value if record.promotion_trace.suppression_reason else ""

    for anchor in sorted_anchors[:_RESCUE_EVIDENCE_LIMIT]:
        before_start = max(0, anchor.start_char - _RESCUE_CONTEXT_RADIUS)
        after_end = min(len(raw_text), anchor.end_char + _RESCUE_CONTEXT_RADIUS)
        context_before = raw_text[before_start:anchor.start_char]
        context_after = raw_text[anchor.end_char:after_end]
        if not context_before.strip() and not context_after.strip():
            continue
        items.append(
            _build_evidence_item(
                source_object_id=record.identity.normalized_key,
                anchor=anchor,
                quote=quote,
                context_before=context_before,
                context_after=context_after,
                suppression_reason=suppression,
                confidence_score=record.promotion_trace.confidence_score,
            )
        )
    return items


@dataclass
class _RescueGroup:
    """Accumulator for merging multiple document-level records of one entity."""

    normalized_key: str
    records: list[DocumentEntityRecord]
    evidence: list[LLMTaskEvidenceItem]
    document_paths: list[str]
    surface_forms: set[str]
    total_occurrences: int
    total_scenes: int
    suppression_reasons: set[str]
    winning_categories: set[str]
    best_confidence: float
    best_entityhood: float


def build_rescue_task_packets(
    bundle: ManuscriptReviewBundle,
    document_texts: dict[str, str],
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build suppression rescue LLM task packets from a manuscript bundle.

    Records sharing the same normalized_key are merged into one packet
    so the LLM sees evidence from all documents in a single call.

    Args:
        bundle: Manuscript review bundle containing entity records.
        document_texts: Map of document path to raw text content, used
            to build evidence windows around rescue candidate anchors.

    Returns:
        Task packets for rescue candidates and selection diagnostics
        for all suppressed records evaluated.
    """
    diagnostics: list[LLMTaskSelectionDiagnostic] = []

    absorbed_keys: frozenset[str] = frozenset(
        source_key
        for entity in bundle.canonical_entities
        for source_key in entity.source_keys
        if source_key != entity.canonical_key
    )

    # First pass: filter and group selected records by normalized_key.
    groups: dict[str, _RescueGroup] = {}

    for record in bundle.entity_records:
        selected, reason = _rescue_candidate_selected(record, absorbed_keys)
        rescue_evidence: list[LLMTaskEvidenceItem] = []

        if selected:
            raw_text = document_texts.get(record.identity.document_anchor.path, "")
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
                source_object_id=record.identity.normalized_key,
                document_path=record.identity.document_anchor.path,
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                selected=selected,
                reason=reason,
                evidence_counts={
                    "occurrence_count": record.source_evidence.occurrence_count,
                    "scene_count": record.promotion_trace.scene_count,
                    "anchor_count": len(record.source_evidence.anchors),
                },
            )
        )
        if not selected:
            continue

        key = record.identity.normalized_key
        if key not in groups:
            groups[key] = _RescueGroup(
                normalized_key=key,
                records=[],
                evidence=[],
                document_paths=[],
                surface_forms=set(),
                total_occurrences=0,
                total_scenes=0,
                suppression_reasons=set(),
                winning_categories=set(),
                best_confidence=0.0,
                best_entityhood=0.0,
            )
        group = groups[key]
        group.records.append(record)
        group.evidence.extend(rescue_evidence)
        if record.identity.document_anchor.path not in group.document_paths:
            group.document_paths.append(record.identity.document_anchor.path)
        group.surface_forms.update(record.identity.surface_forms)
        group.total_occurrences += record.source_evidence.occurrence_count
        group.total_scenes += record.promotion_trace.scene_count
        if record.promotion_trace.suppression_reason is not None:
            group.suppression_reasons.add(record.promotion_trace.suppression_reason.value)
        group.winning_categories.add(record.current_state.winning_category.value)
        group.best_confidence = max(group.best_confidence, record.promotion_trace.confidence_score or 0.0)
        group.best_entityhood = max(group.best_entityhood, record.classification_trace.entityhood.score or 0.0)

    # Second pass: build one packet per deduplicated entity group,
    # capping total evidence to avoid oversized prompts.
    packets: list[LLMTaskPacket] = []
    for group in groups.values():
        capped_evidence = group.evidence[:_RESCUE_EVIDENCE_LIMIT]
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE.value,
                    group.normalized_key,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
                schema_id=_SCHEMA_ID,
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="suppressed_entity_record",
                source_object_id=group.normalized_key,
                source_document_paths=group.document_paths,
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
                evidence_payload=capped_evidence,
                selection_reason="rescue_candidate",
                payload={
                    "normalized_key": group.normalized_key,
                    "surface_forms": sorted(group.surface_forms),
                    "occurrence_count": group.total_occurrences,
                    "scene_count": group.total_scenes,
                    "suppression_reasons": sorted(group.suppression_reasons),
                    "winning_categories": sorted(group.winning_categories),
                    "confidence_score": group.best_confidence,
                    "entityhood_score": group.best_entityhood,
                },
            )
        )

    return packets, diagnostics
