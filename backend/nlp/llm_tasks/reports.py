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
from backend.nlp.types import (
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
    LLMTaskSelectionDiagnostic,
)


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


def _truncate(text: str, *, limit: int = 240) -> str:
    """Return one-line truncated text for compact reports."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def render_llm_task_result_comparison_report(
    packets: list[LLMTaskPacket],
    results: list[LLMTaskResult],
    *,
    max_tasks: int = 24,
    max_evidence_per_task: int = 4,
) -> str:
    """Render deterministic vs LLM side-by-side report for manual review."""
    packet_by_id = {packet.task_id: packet for packet in packets}
    lines: list[str] = []
    lines.append(_hr("LLM RESULT COMPARISON"))
    lines.append(f"  packet_count: {len(packets)}")
    lines.append(f"  result_count: {len(results)}")

    completed = sum(1 for item in results if item.status == LLMTaskResultStatus.COMPLETED)
    failed = sum(1 for item in results if item.status == LLMTaskResultStatus.FAILED)
    skipped = sum(1 for item in results if item.status == LLMTaskResultStatus.SKIPPED)
    lines.append(f"  completed: {completed}")
    lines.append(f"  failed: {failed}")
    lines.append(f"  skipped: {skipped}")

    for result in results[:max_tasks]:
        packet = packet_by_id.get(result.task_id)
        lines.append(_hr(f"TASK {result.task_id}"))
        lines.append(f"  family: {result.task_family.value}")
        lines.append(f"  status: {result.status.value}")
        lines.append(f"  provider: {result.provider}")
        lines.append(f"  model: {result.model}")
        if result.error:
            lines.append(f"  error: {_truncate(result.error)}")
        if packet is None:
            lines.append("  packet: missing")
            continue

        lines.append(f"  source: {packet.source_object_kind}:{packet.source_object_id}")
        lines.append(f"  deterministic_goal: {_truncate(packet.task_goal)}")
        lines.append("  deterministic_constraints:")
        if packet.task_constraints:
            for constraint in packet.task_constraints[:4]:
                lines.append(f"    - {_truncate(constraint, limit=180)}")
        else:
            lines.append("    - None")
        lines.append("  deterministic_evidence:")
        if packet.evidence_payload:
            for item in packet.evidence_payload[:max_evidence_per_task]:
                lines.append(
                    "    - "
                    f"{item.evidence_id} | {item.visibility_bucket} | "
                    f"{_truncate(item.quote, limit=180)}"
                )
        else:
            lines.append("    - None")

        payload = result.payload or {}
        is_valid = bool(payload.get("is_valid", False))
        validation_errors = payload.get("validation_errors", [])
        proposal_payload = payload.get("proposal_payload", {})
        lines.append(f"  llm_validation_valid: {is_valid}")
        if validation_errors:
            lines.append("  llm_validation_errors:")
            for error in validation_errors[:6]:
                lines.append(f"    - {_truncate(str(error), limit=220)}")
        lines.append("  llm_proposal_payload:")
        if proposal_payload:
            for key, value in list(proposal_payload.items())[:12]:
                lines.append(f"    - {key}: {_truncate(str(value), limit=220)}")
        else:
            lines.append("    - None")

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
