"""
LLM task packet report renderer.

.. code-block:: mermaid

    flowchart TD
        A[LLMTaskPacket list] --> C[Render selected task summary]
        B[LLMTaskSelectionDiagnostic list] --> D[Render selection diagnostics]
        C & D --> E[Task-packet inspection report]
"""

from __future__ import annotations

from backend.nlp.text_filtering import strip_emoji
from backend.nlp.types import LLMTaskPacket, LLMTaskSelectionDiagnostic


def _hr(title: str = "") -> str:
    """Return one stable report separator line."""
    width = 72
    if title:
        pad = width - len(title) - 2
        return f"\n-- {title} " + "-" * pad
    return "-" * width


def render_llm_task_packet_report(
    packets: list[LLMTaskPacket],
    diagnostics: list[LLMTaskSelectionDiagnostic],
    *,
    max_packets: int = 8,
) -> str:
    """Render a human-readable task-packet inspection report."""
    lines: list[str] = []
    lines.append(_hr("LLM TASK PACKETS"))
    lines.append(f"  packet_count: {len(packets)}")
    if not packets:
        lines.append("  None.")
    for packet in packets[:max_packets]:
        lines.append(f"  task_id: {packet.task_id}")
        lines.append(f"  task_family: {packet.task_family.value}")
        lines.append(f"  schema_id: {packet.schema_id}")
        lines.append(f"  source_object_id: {packet.source_object_id}")
        lines.append(f"  source_authority: {packet.source_authority}")
        lines.append(f"  source_authority_weight: {packet.source_authority_weight:.2f}")
        lines.append(f"  selection_reason: {packet.selection_reason}")
        lines.append(f"  evidence_count: {len(packet.evidence_payload)}")
        lines.append("")

    lines.append(_hr("TASK SELECTION DIAGNOSTICS"))
    lines.append(f"  diagnostic_count: {len(diagnostics)}")
    if not diagnostics:
        lines.append("  None.")
    for item in diagnostics[: max_packets * 3]:
        lines.append(
            f"  - selected={item.selected} family={item.task_family.value} "
            f"source={item.source_object_kind}:{item.source_object_id} "
            f"reason={item.reason} counts={item.evidence_counts}"
        )

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
