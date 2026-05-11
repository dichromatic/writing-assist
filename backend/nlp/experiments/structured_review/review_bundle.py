"""
Structured-record review bundle builder - packages deterministic record-side evidence.

.. code-block:: mermaid

    flowchart TD
        A[StructuredRecord list] --> B[Keep supported record types]
        C[DocumentEntityRecord list] --> D[Filter overlapping entities]
        E[ReferenceCandidate list] --> F[Filter overlapping references]
        B & D & F --> G[Build DeterministicSeedBundle]
        G --> H[Build RecordReviewBundle]
        H --> I[List of structured review bundles]
"""

from __future__ import annotations

from backend.nlp.experiments.structured_review.prompt_packet import build_record_prompt_packet
from backend.nlp.structured_records.seed_extractor import build_record_seed_bundle
from backend.nlp.types import (
    DocumentEntityRecord,
    RecordReviewBundle,
    ReferenceCandidate,
    StructuredDocumentDiagnostics,
    StructuredRecord,
    StructuredRecordType,
    PendingLLMResponse,
)

_SUPPORTED_RECORD_TYPES = {
    StructuredRecordType.DOSSIER_ENTRY,
    StructuredRecordType.REFERENCE_SECTION,
    StructuredRecordType.OUTLINE_BEAT,
    StructuredRecordType.LOOSE_RECORD,
}


def build_structured_review_bundles(
    records: list[StructuredRecord],
    entity_records: list[DocumentEntityRecord],
    reference_candidates: list[ReferenceCandidate],
) -> tuple[list[RecordReviewBundle], StructuredDocumentDiagnostics]:
    """Build phase-1 structured-note review bundles from structured records.

    Args:
        records: Structured records segmented from one note document.
        entity_records: Whole-document entity summaries used as weak hints.
        reference_candidates: Whole-document deferred references used as weak
            hints.

    Returns:
        Review bundles plus structural diagnostics for the document.
    """
    supported_records = [
        record for record in records
        if record.record_type in _SUPPORTED_RECORD_TYPES
    ]
    diagnostics = StructuredDocumentDiagnostics(
        document_path=records[0].document_path if records else "",
        heading_count=sum(1 for record in records if record.heading_text),
        candidate_record_counts={
            record_type.value: sum(1 for record in records if record.record_type == record_type)
            for record_type in StructuredRecordType
        },
        sample_heading_texts=[
            record.heading_text
            for record in records
            if record.heading_text
        ][:6],
        reason_no_review_bundles=(
            ""
            if supported_records
            else "no supported structured records detected in this document"
        ),
    )
    bundles: list[RecordReviewBundle] = []
    for record in supported_records:
        seed_bundle, subject_guess, fact_candidates = build_record_seed_bundle(
            record,
            entity_records,
            reference_candidates,
        )
        prompt_packet = build_record_prompt_packet(record, seed_bundle, fact_candidates)
        bundles.append(RecordReviewBundle(
            record_id=record.record_id,
            record_type=record.record_type,
            document_path=record.document_path,
            raw_text=record.raw_text,
            llm_prompt_packet=prompt_packet,
            deterministic_seed_bundle=seed_bundle,
            deterministic_subject_guess=subject_guess,
            deterministic_fact_candidates=fact_candidates,
            llm_subject_proposal=PendingLLMResponse(status="not_run_yet"),
            llm_fact_proposals=PendingLLMResponse(status="not_run_yet"),
        ))

    return bundles, diagnostics
