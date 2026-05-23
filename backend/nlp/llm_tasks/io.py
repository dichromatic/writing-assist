"""Input and output helpers for LLM task packet and result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.nlp.text_filtering import to_llm_safe_jsonable
from backend.nlp.types import (
    DocumentStatus,
    DocumentType,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
    LLMTaskSelectionDiagnostic,
    SpanAnchor,
)


def _packet_from_dict(data: dict[str, Any]) -> LLMTaskPacket:
    """Deserialize one LLM task packet from a JSON object."""
    evidence_items = [
        LLMTaskEvidenceItem(
            evidence_id=item["evidence_id"],
            document_path=item["document_path"],
            source_anchor=SpanAnchor(**item["source_anchor"]),
            quote=item["quote"],
            context_before=item.get("context_before", ""),
            context_after=item.get("context_after", ""),
            source_object_id=item["source_object_id"],
            visibility_bucket=item["visibility_bucket"],
            suppression_reason=item.get("suppression_reason", ""),
            confidence_score=item.get("confidence_score"),
            evidence_metadata=(
                dict(item.get("evidence_metadata", {}))
                if isinstance(item.get("evidence_metadata", {}), dict)
                else {}
            ),
        )
        for item in data.get("evidence_payload", [])
    ]
    return LLMTaskPacket(
        task_id=data["task_id"],
        task_family=LLMTaskFamily(data["task_family"]),
        schema_id=data["schema_id"],
        source_bundle_kind=data["source_bundle_kind"],
        source_object_kind=data["source_object_kind"],
        source_object_id=data["source_object_id"],
        source_document_paths=list(data.get("source_document_paths", [])),
        document_type=DocumentType(data["document_type"]),
        document_status=DocumentStatus(data["document_status"]),
        source_authority=data["source_authority"],
        source_authority_weight=float(data["source_authority_weight"]),
        task_goal=data["task_goal"],
        task_constraints=list(data.get("task_constraints", [])),
        evidence_payload=evidence_items,
        selection_reason=data["selection_reason"],
        payload=dict(data.get("payload", {})),
    )


def load_task_packets_from_artifact(path: str) -> list[LLMTaskPacket]:
    """Load LLM task packets from a JSON handoff artifact.

    Args:
        path: Filesystem path to the handoff artifact JSON file.

    Returns:
        Deserialized task packets.
    """
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    packets = content.get("llm_task_packets", [])
    return [_packet_from_dict(item) for item in packets]


def load_task_results_from_artifact(path: str) -> list[LLMTaskResult]:
    """Load LLM task results from a result artifact JSON file.

    Args:
        path: Filesystem path to the result artifact JSON file.

    Returns:
        Deserialized task results.
    """
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    results = content.get("llm_task_results", [])
    parsed: list[LLMTaskResult] = []
    for item in results:
        parsed.append(
            LLMTaskResult(
                task_id=item["task_id"],
                task_family=LLMTaskFamily(item["task_family"]),
                schema_id=item["schema_id"],
                status=LLMTaskResultStatus(item["status"]),
                model=item.get("model", ""),
                provider=item.get("provider", ""),
                response_id=item.get("response_id", ""),
                payload=dict(item.get("payload", {})),
                error=item.get("error", ""),
            )
        )
    return parsed


def write_task_result_artifact(
    *,
    output_path: str,
    source_artifact_paths: list[str],
    packets: list[LLMTaskPacket] | None,
    results: list[LLMTaskResult],
) -> None:
    """Write one JSON artifact containing task packets and results.

    Args:
        output_path: Destination file path.
        source_artifact_paths: Input artifact paths for provenance.
        packets: Task packets included in this run.
        results: Task results from execution.
    """
    payload = {
        "source_artifact_paths": list(source_artifact_paths),
        "llm_task_packets": to_llm_safe_jsonable(packets or []),
        "llm_task_results": to_llm_safe_jsonable(results),
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_handoff_artifact(
    *,
    source_kind: str,
    review_bundle: Any,
    task_packets: list[LLMTaskPacket],
    task_diagnostics: list[LLMTaskSelectionDiagnostic],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready handoff envelope for manuscript artifacts.

    Args:
        source_kind: Source document kind label.
        review_bundle: JSON-safe review bundle data.
        task_packets: Assembled task packets.
        task_diagnostics: Selection diagnostics from assembly.
        extras: Optional extra fields merged into the envelope.

    Returns:
        JSON-serializable handoff artifact.
    """
    artifact: dict[str, Any] = {
        "artifact_version": "2",
        "source_kind": source_kind,
        "review_bundle": to_llm_safe_jsonable(review_bundle),
        "llm_task_packets": to_llm_safe_jsonable(task_packets),
        "llm_task_diagnostics": to_llm_safe_jsonable(task_diagnostics),
    }
    if extras:
        artifact.update(to_llm_safe_jsonable(extras))
    return artifact
