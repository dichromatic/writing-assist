"""
Structured review report renderer - prints deterministic structured-review logs.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle list] --> B[Format deterministic record blocks]
        D[StructuredDocumentDiagnostics] --> E[Format structural summary]
        B & E --> F[Deterministic review report]
"""

from __future__ import annotations

from backend.nlp.report_formatting import hr as _hr
from backend.nlp.text_filtering import strip_emoji
from backend.nlp.types import RecordReviewBundle, StructuredDocumentDiagnostics


def render_structured_review_report(
    diagnostics: StructuredDocumentDiagnostics,
    bundles: list[RecordReviewBundle],
    *,
    max_records: int = 5,
) -> str:
    """Render a human-readable phase-1 structured-note review inspection report.

    Args:
        diagnostics: Structural summary for the source document.
        bundles: Deterministic structured-note review bundles.
        max_records: Maximum number of full record blocks to print.

    Returns:
        Report text for manual inspection.
    """
    lines: list[str] = []
    lines.append(_hr("STRUCTURAL SUMMARY"))
    lines.append(f"  document_path: {diagnostics.document_path}")
    lines.append(f"  document_type: {diagnostics.document_type.value}")
    lines.append(f"  document_status: {diagnostics.document_status.value}")
    lines.append(f"  document_status_source: {diagnostics.document_status_source}")
    if diagnostics.document_status_hints:
        lines.append("  document_status_hints:")
        for hint in diagnostics.document_status_hints:
            lines.append(f"    - {hint}")
    if diagnostics.metadata_conflicts:
        lines.append("  metadata_conflicts:")
        for conflict in diagnostics.metadata_conflicts:
            lines.append(f"    - {conflict}")
    lines.append(f"  heading_count: {diagnostics.heading_count}")
    for record_type, count in sorted(diagnostics.candidate_record_counts.items()):
        lines.append(f"  {record_type}: {count}")
    if diagnostics.sample_heading_texts:
        lines.append("  sample_headings:")
        for heading in diagnostics.sample_heading_texts:
            lines.append(f"    - {heading}")
    if diagnostics.reason_no_review_bundles:
        lines.append(f"  review_status: {diagnostics.reason_no_review_bundles}")

    if not bundles:
        lines.append(_hr("STRUCTURED REVIEW BUNDLES"))
        lines.append("  None.")
        lines.append(_hr())
        return "\n".join(lines) + "\n"

    lines.append(_hr("STRUCTURED REVIEW BUNDLES"))
    for bundle in bundles[:max_records]:
        lines.append(f"  record_id: {bundle.record_id}")
        lines.append(f"  record_type: {bundle.record_type.value}")
        lines.append(f"  document_type: {bundle.document_type.value}")
        lines.append(f"  document_status: {bundle.document_status.value}")
        lines.append(f"  document_path: {bundle.document_path}")
        lines.append(f"  header_line: {bundle.deterministic_seed_bundle.header_line or '-'}")
        lines.append(
            "  suspected_subject_line: "
            f"{bundle.deterministic_seed_bundle.suspected_subject_guess.primary_guess if bundle.deterministic_seed_bundle.suspected_subject_guess else '-'}"
        )
        lines.append(
            "  structural_flags: "
            + (", ".join(bundle.deterministic_seed_bundle.structural_flags) or "-")
        )
        lines.append("  structural_breakdown:")
        for field_line in bundle.deterministic_seed_bundle.field_lines[:12]:
            suffix = (
                f" label={field_line.label!r} value={field_line.value!r}"
                if field_line.label or field_line.value
                else ""
            )
            lines.append(
                f"    - {field_line.line_type.value}: {field_line.raw_text}{suffix}"
            )
        lines.append("  deterministic_candidates:")
        for entity_record in bundle.deterministic_seed_bundle.entity_candidates[:12]:
            if hasattr(entity_record, "normalized_key"):
                lines.append(
                    f"    - entity {entity_record.identity.normalized_key}"
                    f" category={entity_record.current_state.winning_category.value}"
                    f" bucket={entity_record.current_state.bucket.value}"
                    f" conf={entity_record.promotion_trace.confidence_score:.3f}"
                )
            else:
                lines.append(
                    f"    - entity {entity_record.name}"
                    f" normalized={entity_record.normalized_name}"
                    f" source={entity_record.source.value}"
                    f" context={entity_record.source_label or '-'}"
                )
        for reference_candidate in bundle.deterministic_seed_bundle.reference_candidates[:12]:
            lines.append(
                f"    - reference {reference_candidate.normalized}"
                f" type={reference_candidate.reference_type.value}"
                f" links={','.join(reference_candidate.linked_entity_keys) or '-'}"
            )
        lines.append("  deterministic_hints:")
        if bundle.deterministic_subject_guess is not None:
            lines.append(
                f"    - subject_guess={bundle.deterministic_subject_guess.primary_guess}"
                f" reason={bundle.deterministic_subject_guess.reason}"
            )
        if bundle.deterministic_seed_bundle.candidate_rank_texts:
            lines.append(
                "    - header_rank_texts="
                + ", ".join(bundle.deterministic_seed_bundle.candidate_rank_texts)
            )
        if bundle.deterministic_seed_bundle.known_canon_matches:
            lines.append(
                "    - known_canon_matches="
                + ", ".join(bundle.deterministic_seed_bundle.known_canon_matches[:12])
            )
        if bundle.deterministic_fact_candidates:
            lines.append("  deterministic_fact_candidates:")
            for fact_candidate in bundle.deterministic_fact_candidates[:12]:
                lines.append(
                    f"    - {fact_candidate.label}: {fact_candidate.value}"
                    f" reason={fact_candidate.reason}"
                )
        lines.append("  raw_text:")
        for line in bundle.raw_text.splitlines()[:16]:
            lines.append(f"    {line}")
        lines.append("")

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
