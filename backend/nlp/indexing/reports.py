"""Database proposal report helpers."""

from __future__ import annotations

from backend.nlp.text_filtering import strip_emoji
from backend.nlp.types import DatabaseProposal, IndexingDiagnostic


def render_database_proposal_report(
    proposals: list[DatabaseProposal],
    diagnostics: list[IndexingDiagnostic],
) -> str:
    """Render a concise human-readable proposal and validation summary."""
    lines: list[str] = []
    lines.append("DATABASE PROPOSAL SUMMARY")
    lines.append(f"  proposal_count: {len(proposals)}")
    lines.append(f"  diagnostic_count: {len(diagnostics)}")
    insertable = sum(1 for item in proposals if item.insertability_state.value == "insertable")
    not_insertable = sum(1 for item in proposals if item.insertability_state.value == "not_insertable")
    needs_normalization = sum(
        1 for item in proposals if item.insertability_state.value == "needs_normalization"
    )
    lines.append(f"  insertable: {insertable}")
    lines.append(f"  not_insertable: {not_insertable}")
    lines.append(f"  needs_normalization: {needs_normalization}")
    for item in diagnostics[:30]:
        lines.append(
            f"  - [{item.level}] {item.code}"
            f" source={item.source_object_kind}:{item.source_object_id}"
            f" msg={item.message}"
        )
    return strip_emoji("\n".join(lines) + "\n")
