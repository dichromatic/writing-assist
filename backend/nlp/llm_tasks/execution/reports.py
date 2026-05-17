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
    LLMTaskFamily,
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


def _first_present_text(payload: dict, keys: list[str]) -> str:
    """Return the first non-empty text field from payload by key order."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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
    triage_payloads = [
        ((item.payload or {}).get("proposal_payload", {}))
        for item in results
        if item.status == LLMTaskResultStatus.COMPLETED
    ]
    passing_count = sum(1 for payload in triage_payloads if payload.get("passing") is True)
    failing_count = sum(1 for payload in triage_payloads if payload.get("failing") is True)
    lines.append(f"  passing_count: {passing_count}")
    lines.append(f"  failing_count: {failing_count}")

    for result in results[:max_tasks]:
        packet = packet_by_id.get(result.task_id)
        lines.append(_hr(f"TASK {result.task_id}"))
        lines.append(f"  family: {result.task_family.value}")
        lines.append(f"  pass_stage: {result.pass_stage.value}")
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
        rationale = _first_present_text(
            proposal_payload,
            [
                "resolution_rationale",
                "remaining_uncertainty",
                "uncertainty_reason",
                "rationale",
                "justification",
                "resolution_notes",
                "profile_summary",
                "notes",
                "profile_notes",
            ],
        )
        if rationale:
            lines.append("  llm_rationale:")
            lines.append(f"    {_truncate(rationale, limit=1200)}")

    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")


def render_structured_tagged_extraction_report(
    packets: list[LLMTaskPacket],
    results: list[LLMTaskResult],
    *,
    max_tasks: int = 40,
    max_items_per_task: int = 12,
) -> str:
    """Render a focused report for structured tagged extraction task outputs."""
    packet_by_id = {packet.task_id: packet for packet in packets}
    tagged_results = [
        item
        for item in results
        if item.task_family == LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION
    ]
    lines: list[str] = []
    lines.append(_hr("STRUCTURED TAGGED EXTRACTION"))
    lines.append(f"  tagged_task_count: {len(tagged_results)}")
    if not tagged_results:
        lines.append("  None.")
        lines.append(_hr())
        return strip_emoji("\n".join(lines) + "\n")

    completed = sum(1 for item in tagged_results if item.status == LLMTaskResultStatus.COMPLETED)
    failed = sum(1 for item in tagged_results if item.status == LLMTaskResultStatus.FAILED)
    skipped = sum(1 for item in tagged_results if item.status == LLMTaskResultStatus.SKIPPED)
    lines.append(f"  completed: {completed}")
    lines.append(f"  failed: {failed}")
    lines.append(f"  skipped: {skipped}")

    for result in tagged_results[:max_tasks]:
        packet = packet_by_id.get(result.task_id)
        lines.append(_hr(f"TASK {result.task_id}"))
        lines.append(f"  status: {result.status.value}")
        lines.append(f"  model: {result.model}")
        if packet is not None:
            lines.append(f"  source_record: {packet.source_object_id}")
            lines.append(f"  record_type: {packet.payload.get('record_type', '')}")
            lines.append(f"  document_path: {packet.source_document_paths[0] if packet.source_document_paths else ''}")
        if result.error:
            lines.append(f"  error: {_truncate(result.error)}")

        envelope = result.payload or {}
        is_valid = bool(envelope.get("is_valid", False))
        lines.append(f"  valid: {is_valid}")
        errors = envelope.get("validation_errors", [])
        if errors:
            lines.append("  validation_errors:")
            for error in errors[:6]:
                lines.append(f"    - {_truncate(str(error), limit=220)}")

        proposal_payload = envelope.get("proposal_payload", {})
        items = proposal_payload.get("extraction_items", [])
        lines.append(f"  extraction_item_count: {len(items)}")
        for item in items[:max_items_per_task]:
            type_tag = item.get("type_tag", "")
            subject_names = item.get("subject_names", [])
            subject_text = ", ".join(subject_names) if isinstance(subject_names, list) else str(subject_names)
            content = _truncate(str(item.get("content", "")), limit=260)
            evidence_quote = _truncate(str(item.get("evidence_quote", "")), limit=220)
            lines.append(f"    - type={type_tag} subjects=[{subject_text}]")
            lines.append(f"      content={content}")
            lines.append(f"      evidence={evidence_quote}")
    lines.append(_hr())
    return strip_emoji("\n".join(lines) + "\n")
