"""
LLM task report renderers for packet inspection and result comparison.

.. code-block:: mermaid

    flowchart TD
        A[LLMTaskPacket list] --> B[Render packet summary]
        C[LLMTaskSelectionDiagnostic list] --> D[Render selection diagnostics]
        E[LLMTaskResult list] --> F[Render result comparison]
        B & D --> G[Packet inspection report]
        A & F --> H[Result comparison report]
"""

from __future__ import annotations

from backend.nlp.report_formatting import hr as _hr
from backend.nlp.text_filtering import strip_emoji
from backend.nlp.types import (
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
    LLMTaskSelectionDiagnostic,
)


def render_task_packet_report(
    packets: list[LLMTaskPacket],
    diagnostics: list[LLMTaskSelectionDiagnostic],
    *,
    max_packets: int = 8,
) -> str:
    """Render a human-readable task-packet inspection report.

    Args:
        packets: Assembled task packets.
        diagnostics: Selection diagnostics from assembly.
        max_packets: Maximum number of packet details to include.

    Returns:
        Formatted text report.
    """
    lines: list[str] = []
    lines.append(_hr("LLM TASK PACKETS"))
    lines.append(f"  packet_count: {len(packets)}")

    if not packets:
        lines.append("  None.")
    for packet in packets[:max_packets]:
        lines.append(f"  task_id: {packet.task_id}")
        lines.append(f"  source_object_id: {packet.source_object_id}")
        lines.append(f"  selection_reason: {packet.selection_reason}")
        lines.append(f"  evidence_count: {len(packet.evidence_payload)}")
        lines.append("")

    lines.append(_hr("TASK SELECTION DIAGNOSTICS"))
    lines.append(f"  diagnostic_count: {len(diagnostics)}")
    selected_count = sum(1 for d in diagnostics if d.selected)
    rejected_count = len(diagnostics) - selected_count
    lines.append(f"  selected: {selected_count}")
    lines.append(f"  rejected: {rejected_count}")

    if not diagnostics:
        lines.append("  None.")
    for item in diagnostics[:max_packets * 3]:
        lines.append(
            f"  - selected={item.selected}"
            f"  source={item.source_object_id}"
            f"  reason={item.reason}"
            f"  counts={item.evidence_counts}"
        )

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")


def _truncate(text: str, *, limit: int = 240) -> str:
    """Return one-line truncated text for compact reports."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def render_task_result_report(
    packets: list[LLMTaskPacket],
    results: list[LLMTaskResult],
    *,
    max_tasks: int = 60,
) -> str:
    """Render a rescue result report for manual review.

    Args:
        packets: Task packets that were executed.
        results: Execution results from the provider.
        max_tasks: Maximum number of task details to include.

    Returns:
        Formatted text report with verdict summaries.
    """
    packet_by_id = {p.task_id: p for p in packets}
    lines: list[str] = []
    lines.append(_hr("RESCUE RESULTS"))
    lines.append(f"  packet_count: {len(packets)}")
    lines.append(f"  result_count: {len(results)}")

    completed = sum(1 for r in results if r.status == LLMTaskResultStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == LLMTaskResultStatus.FAILED)
    skipped = sum(1 for r in results if r.status == LLMTaskResultStatus.SKIPPED)
    lines.append(f"  completed: {completed}")
    lines.append(f"  failed: {failed}")
    lines.append(f"  skipped: {skipped}")

    rescued_count = 0
    suppressed_count = 0
    for result in results:
        if result.status != LLMTaskResultStatus.COMPLETED:
            continue
        proposal = (result.payload or {}).get("proposal_payload", {})
        if proposal.get("rescue") is True:
            rescued_count += 1
        else:
            suppressed_count += 1
    lines.append(f"  rescued: {rescued_count}")
    lines.append(f"  correctly_suppressed: {suppressed_count}")

    for result in results[:max_tasks]:
        packet = packet_by_id.get(result.task_id)
        source_id = packet.source_object_id if packet else result.task_id
        lines.append(_hr(source_id))
        lines.append(f"  status: {result.status.value}")
        if result.error:
            lines.append(f"  error: {_truncate(result.error)}")
            continue

        if packet is not None:
            payload = packet.payload
            lines.append(
                f"  suppression_reason: {payload.get('suppression_reason', '')}"
                f"  occurrences: {payload.get('occurrence_count', '')}"
                f"  scenes: {payload.get('scene_count', '')}"
            )

        proposal = (result.payload or {}).get("proposal_payload", {})
        rescue = proposal.get("rescue")
        lines.append(f"  rescue: {rescue}")
        if rescue:
            lines.append(f"  entity_type: {proposal.get('entity_type', '')}")
            lines.append(f"  canonical_name: {proposal.get('canonical_name', '')}")
        confidence = proposal.get("confidence")
        if confidence is not None:
            lines.append(f"  confidence: {confidence}")
        rationale = proposal.get("rationale", "")
        if rationale:
            lines.append(f"  rationale: {_truncate(rationale)}")

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
