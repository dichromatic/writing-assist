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
from backend.nlp.llm_tasks.assembly.evidence import build_evidence_item
from backend.nlp.llm_tasks.assembly.schemas import schema_id_for
from backend.nlp.types import (
    DeterministicFactCandidate,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskSelectionDiagnostic,
    RecordReviewBundle,
    StructuredEntityMention,
    stable_hash_id,
)


def _entity_evidence(
    bundle: RecordReviewBundle,
    entity_candidates: list[StructuredEntityMention],
) -> list[LLMTaskEvidenceItem]:
    """Convert deterministic entity candidates into bounded evidence items."""
    items: list[LLMTaskEvidenceItem] = []
    for candidate in entity_candidates:
        items.append(
            build_evidence_item(
                source_object_id=bundle.record_id,
                anchor=candidate.anchor,
                quote=candidate.name,
                visibility_bucket=f"structured_entity:{candidate.source.value}",
                confidence_score=None,
            )
        )
    return items


def _fact_evidence(bundle: RecordReviewBundle) -> list[LLMTaskEvidenceItem]:
    """Convert deterministic fact candidates into bounded evidence items."""
    items: list[LLMTaskEvidenceItem] = []
    for fact in bundle.deterministic_fact_candidates:
        items.append(
            build_evidence_item(
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


def _tagged_extraction_constraints(record_type: str) -> list[str]:
    """Return record-type-specific constraints for tagged extraction tasks."""
    constraints = [
        "Use only one of the allowed type_tag values.",
        "Ground each item in a short quoted evidence span from this record.",
        "Do not force unresolved semantics into a hard category.",
        "Do not emit section headings or structural metadata as entity_mention items.",
    ]
    if record_type == "dossier_entry":
        constraints.append(
            "When a dossier subject is explicit in header hints, include at least one entity_mention for that subject."
        )
    elif record_type == "outline_beat":
        constraints.append(
            "Prefer event_description for action-oriented beats and planned sequences."
        )
    elif record_type == "loose_record":
        constraints.append(
            "Use unclassified more liberally when meaning is useful but entity/event framing is uncertain."
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


def build_structured_record_tagged_extraction_task_packets(
    bundles: list[RecordReviewBundle],
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build tagged extraction packets for per-record semantic extraction."""
    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []
    for bundle in bundles:
        field_lines_payload = [
            {
                "line_index": line.line_index,
                "line_type": line.line_type.value,
                "raw_text": line.raw_text,
                "label": line.label,
                "value": line.value,
            }
            for line in bundle.deterministic_seed_bundle.field_lines
        ]
        stage1_hints = []
        for candidate in bundle.deterministic_seed_bundle.entity_candidates:
            if isinstance(candidate, StructuredEntityMention):
                stage1_hints.append(
                    {
                        "name": candidate.name,
                        "normalized_name": candidate.normalized_name,
                        "source": candidate.source.value,
                        "source_label": candidate.source_label,
                    }
                )
        evidence_items = _fact_evidence(bundle)
        evidence_items.extend(_entity_evidence(bundle, bundle.deterministic_seed_bundle.entity_candidates))

        if not bundle.raw_text.strip():
            diagnostics.append(
                LLMTaskSelectionDiagnostic(
                    source_bundle_kind="record_review_bundle",
                    source_object_kind="structured_record",
                    source_object_id=bundle.record_id,
                    document_path=bundle.document_path,
                    task_family=LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION,
                    selected=False,
                    reason="empty_record_text",
                    evidence_counts={
                        "fact_candidates": len(bundle.deterministic_fact_candidates),
                        "entity_candidates": len(bundle.deterministic_seed_bundle.entity_candidates),
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
                task_family=LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION,
                selected=True,
                reason="record_has_structural_context",
                evidence_counts={
                    "fact_candidates": len(bundle.deterministic_fact_candidates),
                    "entity_candidates": len(bundle.deterministic_seed_bundle.entity_candidates),
                    "field_lines": len(bundle.deterministic_seed_bundle.field_lines),
                },
            )
        )
        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION.value,
                    bundle.record_id,
                    bundle.document_path,
                ),
                task_family=LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION,
                schema_id=schema_id_for(LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION),
                source_bundle_kind="record_review_bundle",
                source_object_kind="structured_record",
                source_object_id=bundle.record_id,
                source_document_paths=[bundle.document_path],
                document_type=bundle.document_type,
                document_status=bundle.document_status,
                source_authority=f"structured_record:{bundle.record_type.value}",
                source_authority_weight=document_status_authority_weight(bundle.document_status),
                task_goal=(
                    "Extract tagged semantic items from one structured record using"
                    " structural context and evidence-anchored quotes."
                ),
                task_constraints=[
                    *_tagged_extraction_constraints(bundle.record_type.value),
                ],
                evidence_payload=evidence_items,
                selection_reason="record_has_structural_context",
                payload={
                    "record_type": bundle.record_type.value,
                    "raw_record_text": bundle.raw_text,
                    "header_line": bundle.deterministic_seed_bundle.header_line,
                    "document_type": bundle.document_type.value,
                    "document_status": bundle.document_status.value,
                    "field_lines": field_lines_payload,
                    "stage1_entity_hints": stage1_hints,
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
                        for item in bundle.deterministic_fact_candidates
                    ],
                },
            )
        )
    return packets, diagnostics
