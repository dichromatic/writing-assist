"""
Dossier review report renderer - prints deterministic and LLM dossier logs.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle list] --> B[Format deterministic record blocks]
        A --> C[Format LLM review blocks]
        D[StructuredDocumentDiagnostics] --> E[Format structural summary]
        B & E --> F[Deterministic review report]
        C & E --> G[LLM review report]
"""

from __future__ import annotations

from backend.nlp.text_filtering import strip_emoji
from backend.nlp.types import RecordReviewBundle, StructuredDocumentDiagnostics


def _hr(title: str = "") -> str:
    """Return one stable report separator line.

    Args:
        title: Optional section title.

    Returns:
        Separator line for the report.
    """
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def render_dossier_review_report(
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
    lines.append(f"  heading_count: {diagnostics.heading_count}")
    for record_type, count in sorted(diagnostics.candidate_record_counts.items()):
        lines.append(f"  {record_type}: {count}")
    if diagnostics.sample_heading_texts:
        lines.append("  sample_headings:")
        for heading in diagnostics.sample_heading_texts:
            lines.append(f"    - {heading}")
    if diagnostics.reason_no_dossier_bundles:
        lines.append(f"  review_status: {diagnostics.reason_no_dossier_bundles}")

    if not bundles:
        lines.append(_hr("STRUCTURED REVIEW BUNDLES"))
        lines.append("  None.")
        lines.append(_hr())
        return "\n".join(lines) + "\n"

    lines.append(_hr("STRUCTURED REVIEW BUNDLES"))
    for bundle in bundles[:max_records]:
        lines.append(f"  record_id: {bundle.record_id}")
        lines.append(f"  record_type: {bundle.record_type.value}")
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
            lines.append(
                f"    - entity {entity_record.normalized_key}"
                f" category={entity_record.winning_category.value}"
                f" bucket={entity_record.bucket.value}"
                f" conf={entity_record.confidence_score:.3f}"
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


def render_dossier_llm_report(
    diagnostics: StructuredDocumentDiagnostics,
    bundles: list[RecordReviewBundle],
    *,
    max_records: int = 5,
) -> str:
    """Render the separate LLM-side dossier review log.

    Args:
        diagnostics: Structural summary for the source document.
        bundles: Structured-note review bundles, including deterministic and LLM slots.
        max_records: Maximum number of full record blocks to print.

    Returns:
        LLM-focused report text for manual inspection.
    """
    lines: list[str] = []
    lines.append(_hr("STRUCTURAL SUMMARY"))
    lines.append(f"  document_path: {diagnostics.document_path}")
    lines.append(f"  heading_count: {diagnostics.heading_count}")
    for record_type, count in sorted(diagnostics.candidate_record_counts.items()):
        lines.append(f"  {record_type}: {count}")

    if not bundles:
        lines.append(_hr("STRUCTURED LLM REVIEW"))
        lines.append("  None.")
        lines.append(_hr())
        return "\n".join(lines) + "\n"

    lines.append(_hr("STRUCTURED LLM REVIEW"))
    for bundle in bundles[:max_records]:
        lines.append(f"  record_id: {bundle.record_id}")
        lines.append(f"  record_type: {bundle.record_type.value}")
        lines.append(f"  document_path: {bundle.document_path}")
        lines.append(f"  header_line: {bundle.deterministic_seed_bundle.header_line or '-'}")
        lines.append(f"  llm_task: {bundle.llm_prompt_packet.task_name}")
        lines.append(f"  source_authority: {bundle.llm_prompt_packet.source_authority}")
        lines.append(f"  task_goal: {bundle.llm_prompt_packet.task_goal}")
        lines.append("  task_constraints:")
        for constraint in bundle.llm_prompt_packet.task_constraints:
            lines.append(f"    - {constraint}")

        lines.append(f"  llm_subject_proposal: {bundle.llm_subject_proposal.status}")
        if bundle.llm_subject_proposal.status == "completed":
            payload = bundle.llm_subject_proposal.payload
            lines.append(f"    subject_name: {payload.get('subject_name', '-')}")
            alternate_names = payload.get("alternate_names", [])
            evidence_quotes = payload.get("evidence_quotes", [])
            lines.append(
                "    alternate_names: "
                + (", ".join(alternate_names) if alternate_names else "-")
            )
            lines.append(f"    certainty_note: {payload.get('certainty_note', '-')}")
            lines.append(f"    unresolved: {payload.get('unresolved', False)}")
            if evidence_quotes:
                lines.append("    evidence_quotes:")
                for quote in evidence_quotes[:5]:
                    lines.append(f"      - {quote}")
        elif bundle.llm_subject_proposal.status == "failed":
            lines.append(f"    error: {bundle.llm_subject_proposal.error}")

        lines.append(f"  llm_fact_proposals: {bundle.llm_fact_proposals.status}")
        if bundle.llm_fact_proposals.status == "completed":
            fact_items = bundle.llm_fact_proposals.payload.get("items", [])
            for fact_item in fact_items[:12]:
                lines.append(
                    f"    - {fact_item.get('label', '-')}: {fact_item.get('value', '-')}"
                    f" evidence={fact_item.get('evidence_quote', '-')}"
                    f" note={fact_item.get('certainty_note', '-')}"
                )
        elif bundle.llm_fact_proposals.status == "failed":
            lines.append(f"    error: {bundle.llm_fact_proposals.error}")

        if bundle.agreement_items:
            lines.append("  agreement_items:")
            for item in bundle.agreement_items[:12]:
                lines.append(f"    - {item}")
        if bundle.deterministic_only_items:
            lines.append("  deterministic_only_items:")
            for item in bundle.deterministic_only_items[:12]:
                lines.append(f"    - {item}")
        if bundle.llm_only_items:
            lines.append("  llm_only_items:")
            for item in bundle.llm_only_items[:12]:
                lines.append(f"    - {item}")
        if bundle.open_questions:
            lines.append("  open_questions:")
            for item in bundle.open_questions[:12]:
                lines.append(f"    - {item}")
        lines.append("")

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
