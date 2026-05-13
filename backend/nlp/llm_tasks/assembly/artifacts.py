"""Shared artifact envelope helpers for review-bundle to task-packet handoff."""

from __future__ import annotations

from typing import Any

from backend.nlp.text_filtering import to_llm_safe_jsonable
from backend.nlp.types import LLMTaskPacket, LLMTaskSelectionDiagnostic

REVIEW_BUNDLE_ARTIFACT_VERSION = "1"


def build_review_bundle_handoff_artifact(
    *,
    source_kind: str,
    review_bundle_kind: str,
    review_bundle: Any,
    llm_task_packets: list[LLMTaskPacket],
    llm_task_diagnostics: list[LLMTaskSelectionDiagnostic],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a shared JSON-ready handoff envelope for both source families."""
    payload: dict[str, Any] = {
        "review_bundle_artifact_version": REVIEW_BUNDLE_ARTIFACT_VERSION,
        "source_kind": source_kind,
        "review_bundle_kind": review_bundle_kind,
        "review_bundle": to_llm_safe_jsonable(review_bundle),
        "llm_task_packets": to_llm_safe_jsonable(llm_task_packets),
        "llm_task_diagnostics": to_llm_safe_jsonable(llm_task_diagnostics),
    }
    if extras:
        payload.update(to_llm_safe_jsonable(extras))
    return payload
