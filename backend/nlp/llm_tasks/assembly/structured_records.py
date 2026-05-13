"""
Structured-record LLM task builder - deterministic review bundle to shared task packets.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle list] --> B[Evaluate selection rules]
        B --> C{Selected}
        C -->|Yes| D[Build LLMTaskPacket]
        C -->|No| E[Build LLMTaskSelectionDiagnostic]
        D --> F[Packet list]
        E --> G[Diagnostic list]
"""

from __future__ import annotations

from backend.nlp.document_metadata import document_status_authority_weight
from backend.nlp.llm_tasks.assembly.schemas import schema_id_for
from backend.nlp.types import (
    DeterministicFactCandidate,
    DocumentEntityRecord,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskSelectionDiagnostic,
    RecordReviewBundle,
    SpanAnchor,
    stable_hash_id,
)


def _evidence_item(
    *,
    source_object_id: str,
    anchor: SpanAnchor,
    quote: str,
    visibility_bucket: str,
    suppression_reason: str = "",
    confidence_score: float | None = None,
) -> LLMTaskEvidenceItem:
    """Build one bounded evidence item for a task packet."""
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
        context_before="",
        context_after="",
        source_object_id=source_object_id,
        visibility_bucket=visibility_bucket,
        suppression_reason=suppression_reason,
        confidence_score=confidence_score,
    )


def _entity_evidence(
    bundle: RecordReviewBundle,
    entity_candidates: list[DocumentEntityRecord],
) -> list[LLMTaskEvidenceItem]:
    """Convert deterministic entity candidates into bounded evidence items."""
    items: list[LLMTaskEvidenceItem] = []
    for candidate in entity_candidates:
        items.append(
            _evidence_item(
                source_object_id=bundle.record_id,
                anchor=candidate.anchors[0],
                quote=(candidate.surface_forms[0] if candidate.surface_forms else candidate.normalized_key),
                visibility_bucket=candidate.bucket.value,
                suppression_reason=(
                    candidate.suppression_reason.value
                    if candidate.suppression_reason is not None
                    else ""
                ),
                confidence_score=candidate.confidence_score,
            )
        )
    return items


def _fact_evidence(bundle: RecordReviewBundle) -> list[LLMTaskEvidenceItem]:
    """Convert deterministic fact candidates into bounded evidence items."""
    items: list[LLMTaskEvidenceItem] = []
    for fact in bundle.deterministic_fact_candidates:
        items.append(
                _evidence_item(
                    source_object_id=bundle.record_id,
                    anchor=fact.supporting_anchor,
                    quote=f"{fact.label}: {fact.value}",
                    visibility_bucket="deterministic_fact_candidate",
                )
            )
    return items


def _task_constraints(record_type: str) -> list[str]:
    """Return conservative constraints for record-fact extraction."""
    constraints = [
        "Extract evidence-backed proposals only.",
        "Keep unresolved subjects unresolved when evidence is ambiguous.",
        "Do not infer canon authority from writing style.",
        "Do not extract section headings, record labels, or structural metadata as lore facts.",
        "Each fact must be atomic and testable. Do not parrot full paragraphs.",
        f"Record family is {record_type}; preserve this context in the output.",
    ]
    if record_type == "loose_record":
        constraints.append(
            "For loose_record prose, decompose into multiple atomic facts when multiple claims are present."
        )
    return constraints


def build_structured_record_task_packets(
    bundles: list[RecordReviewBundle],
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build shared LLM task packets for structured-record review bundles."""
    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []
    for bundle in bundles:
        entity_candidates = bundle.deterministic_seed_bundle.entity_candidates
        fact_candidates: list[DeterministicFactCandidate] = bundle.deterministic_fact_candidates
        evidence_items = _fact_evidence(bundle)
        evidence_items.extend(_entity_evidence(bundle, entity_candidates))
        if not fact_candidates and not entity_candidates:
            diagnostics.append(
                LLMTaskSelectionDiagnostic(
                    source_bundle_kind="record_review_bundle",
                    source_object_kind="structured_record",
                    source_object_id=bundle.record_id,
                    document_path=bundle.document_path,
                    task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
                    selected=False,
                    reason="no_extraction_evidence",
                    evidence_counts={
                        "fact_candidates": len(fact_candidates),
                        "entity_candidates": len(entity_candidates),
                    },
                )
            )
            continue

        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="record_review_bundle",
                source_object_kind="structured_record",
                source_object_id=bundle.record_id,
                document_path=bundle.document_path,
                task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
                selected=True,
                reason="record_has_fact_candidates",
                evidence_counts={
                    "fact_candidates": len(fact_candidates),
                    "entity_candidates": len(entity_candidates),
                    "reference_candidates": len(bundle.deterministic_seed_bundle.reference_candidates),
                },
            )
        )
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.RECORD_FACT_EXTRACTION.value,
                    bundle.record_id,
                    bundle.document_path,
                ),
                task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
                schema_id=schema_id_for(LLMTaskFamily.RECORD_FACT_EXTRACTION),
                source_bundle_kind="record_review_bundle",
                source_object_kind="structured_record",
                source_object_id=bundle.record_id,
                source_document_paths=[bundle.document_path],
                document_type=bundle.document_type,
                document_status=bundle.document_status,
                source_authority=f"structured_record:{bundle.record_type.value}",
                source_authority_weight=document_status_authority_weight(bundle.document_status),
                task_goal=(
                    "Extract proposition-centered, evidence-backed record facts "
                    "without forcing unsupported subject resolution."
                ),
                task_constraints=_task_constraints(bundle.record_type.value),
                evidence_payload=evidence_items,
                selection_reason="record_has_fact_candidates",
                payload={
                    "record_type": bundle.record_type.value,
                    "raw_record_text": bundle.raw_text,
                    "header_line": bundle.deterministic_seed_bundle.header_line,
                    "deterministic_subject_guess": (
                        bundle.deterministic_subject_guess.primary_guess
                        if bundle.deterministic_subject_guess
                        else ""
                    ),
                    "deterministic_fact_candidates": [
                        {
                            "label": item.label,
                            "value": item.value,
                            "reason": item.reason,
                            "line_index": item.line_index,
                        }
                        for item in fact_candidates
                    ],
                },
            )
        )
    return packets, diagnostics
