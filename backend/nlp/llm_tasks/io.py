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
    SpanAnchor,
)


def _packet_from_dict(payload: dict[str, Any]) -> LLMTaskPacket:
    """Build an LLM task packet from one JSON object."""
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
        )
        for item in payload.get("evidence_payload", [])
    ]
    return LLMTaskPacket(
        task_id=payload["task_id"],
        task_family=LLMTaskFamily(payload["task_family"]),
        schema_id=payload["schema_id"],
        source_bundle_kind=payload["source_bundle_kind"],
        source_object_kind=payload["source_object_kind"],
        source_object_id=payload["source_object_id"],
        source_document_paths=list(payload.get("source_document_paths", [])),
        document_type=DocumentType(payload["document_type"]),
        document_status=DocumentStatus(payload["document_status"]),
        source_authority=payload["source_authority"],
        source_authority_weight=float(payload["source_authority_weight"]),
        task_goal=payload["task_goal"],
        task_constraints=list(payload.get("task_constraints", [])),
        evidence_payload=evidence_items,
        selection_reason=payload["selection_reason"],
        payload=dict(payload.get("payload", {})),
    )


def load_task_packets_from_artifact(path: str) -> list[LLMTaskPacket]:
    """Load shared LLM task packets from one JSON handoff artifact."""
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    packets = content.get("llm_task_packets", [])
    return [_packet_from_dict(item) for item in packets]


def write_task_result_artifact(
    *,
    output_path: str,
    source_artifact_paths: list[str],
    results: list[LLMTaskResult],
) -> None:
    """Write one JSON artifact containing task results."""
    payload = {
        "source_artifact_paths": list(source_artifact_paths),
        "llm_task_results": to_llm_safe_jsonable(results),
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_task_results_from_artifact(path: str) -> list[LLMTaskResult]:
    """Load shared LLM task results from one result artifact JSON file."""
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
