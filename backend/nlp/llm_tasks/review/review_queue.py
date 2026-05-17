"""
Review queue builder - select deferred manuscript entities for second-pass resolution.

.. code-block:: mermaid

    flowchart TD
        A[LLMTaskPacket list] --> C[Index packets by task_id]
        B[LLMTaskResult list] --> D[Filter deferred manuscript results]
        C & D --> E[Join result + packet evidence]
        E --> F[Diversity-first snippet selection]
        F --> G[Review queue artifact items]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.nlp.parsing.document_parser import parse as parse_document
from backend.nlp.types import LLMTaskFamily, LLMTaskPacket, LLMTaskResult, LLMTaskResultStatus


@dataclass(frozen=True)
class ReviewQueueItem:
    """One deferred manuscript entity queued for second-pass resolution."""

    queue_id: str
    canonical_key: str
    task_id: str
    source_document_paths: list[str]
    deterministic_prior: dict[str, Any]
    first_pass_assessment: dict[str, Any]
    evidence_snippets: list[dict[str, Any]]
    max_context_chars: int


def _proposal_payload(result: LLMTaskResult) -> dict[str, Any]:
    """Return normalized proposal payload from one task result."""
    payload = result.payload or {}
    if not isinstance(payload, dict):
        return {}
    proposal = payload.get("proposal_payload", {})
    return proposal if isinstance(proposal, dict) else {}


def _needs_second_pass(proposal: dict[str, Any]) -> bool:
    """Return true when first-pass triage marks the entity as unresolved."""
    passing = proposal.get("passing")
    if isinstance(passing, bool):
        return not passing
    return True


def _snippet_from_evidence_item(item) -> dict[str, Any]:
    """Convert one packet evidence item into queue snippet shape."""
    return {
        "evidence_id": item.evidence_id,
        "document_path": item.document_path,
        "span_ordinal": item.source_anchor.span_ordinal,
        "start_char": item.source_anchor.start_char,
        "end_char": item.source_anchor.end_char,
        "quote": item.quote,
        "context_before": item.context_before,
        "context_after": item.context_after,
        "confidence_score": item.confidence_score,
        "visibility_bucket": item.visibility_bucket,
    }


def _scene_index_by_span(path: str) -> dict[int, dict[str, int]]:
    """Build span-ordinal to scene metadata index for one source document."""
    try:
        raw_text = open(path, encoding="utf-8").read()
    except OSError:
        return {}
    parsed = parse_document(path, raw_text)
    index: dict[int, dict[str, int]] = {}
    for scene in parsed.scenes:
        for ordinal in scene.span_ordinals:
            index[ordinal] = {
                "scene_index": int(scene.scene_index),
                "scene_start": int(scene.start_char),
                "scene_end": int(scene.end_char),
            }
    return index


def _bounded_scene_excerpt(
    *,
    raw_text: str,
    scene_start: int,
    scene_end: int,
    anchor_start: int,
    anchor_end: int,
    max_context_chars: int,
) -> str:
    """Return a bounded scene excerpt centered on the mention anchor."""
    scene_text = raw_text[scene_start:scene_end]
    if len(scene_text) <= max_context_chars:
        return scene_text

    mention_center = (anchor_start + anchor_end) // 2
    local_center = max(0, mention_center - scene_start)
    half = max_context_chars // 2
    clip_start = max(0, local_center - half)
    clip_end = min(len(scene_text), clip_start + max_context_chars)
    if clip_end - clip_start < max_context_chars:
        clip_start = max(0, clip_end - max_context_chars)
    return scene_text[clip_start:clip_end]


def _select_diverse_snippets(
    packet: LLMTaskPacket,
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    """Return diversity-first snippets for one manuscript entity packet."""
    snippets = [_snippet_from_evidence_item(item) for item in packet.evidence_payload]
    if not snippets:
        return []

    for snippet in snippets:
        if snippet["confidence_score"] is None:
            snippet["confidence_score"] = 0.0

    # Primary diversity key is document, then span ordinal; confidence is
    # secondary ranking within each group.
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for snippet in snippets:
        key = (snippet["document_path"], int(snippet["span_ordinal"]))
        grouped.setdefault(key, []).append(snippet)
    for group in grouped.values():
        group.sort(key=lambda item: float(item.get("confidence_score", 0.0)), reverse=True)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: float(group[0].get("confidence_score", 0.0)),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    group_index = 0
    while len(selected) < max_snippets and ordered_groups:
        group = ordered_groups[group_index % len(ordered_groups)]
        if group:
            selected.append(group.pop(0))
        ordered_groups = [item for item in ordered_groups if item]
        group_index += 1
    return selected


def build_manuscript_review_queue(
    *,
    task_packets: list[LLMTaskPacket],
    task_results: list[LLMTaskResult],
    max_snippets: int = 5,
    max_context_chars: int = 2000,
) -> list[ReviewQueueItem]:
    """Build deferred-entity queue items from first-pass manuscript outputs."""
    packet_by_id = {packet.task_id: packet for packet in task_packets}
    scene_index_cache: dict[str, dict[int, dict[str, int]]] = {}
    raw_text_cache: dict[str, str] = {}
    queue_items: list[ReviewQueueItem] = []
    seen_keys: set[str] = set()

    for result in task_results:
        if result.status != LLMTaskResultStatus.COMPLETED:
            continue
        if result.task_family != LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
            continue
        proposal = _proposal_payload(result)
        if not _needs_second_pass(proposal):
            continue
        canonical_key = str(
            proposal.get("canonical_key")
            or proposal.get("entity_key")
            or proposal.get("canonical_name")
            or ""
        ).strip()
        if not canonical_key or canonical_key in seen_keys:
            continue
        packet = packet_by_id.get(result.task_id)
        if packet is None:
            continue
        seen_keys.add(canonical_key)
        snippets = _select_diverse_snippets(packet, max_snippets=max_snippets)
        enriched_snippets: list[dict[str, Any]] = []
        for snippet in snippets:
            path = str(snippet.get("document_path", ""))
            span_ordinal = int(snippet.get("span_ordinal", 0))
            if path and path not in scene_index_cache:
                scene_index_cache[path] = _scene_index_by_span(path)
            if path and path not in raw_text_cache:
                try:
                    raw_text_cache[path] = open(path, encoding="utf-8").read()
                except OSError:
                    raw_text_cache[path] = ""

            scene_meta = scene_index_cache.get(path, {}).get(span_ordinal)
            if scene_meta is not None and path in raw_text_cache and raw_text_cache[path]:
                excerpt = _bounded_scene_excerpt(
                    raw_text=raw_text_cache[path],
                    scene_start=int(scene_meta["scene_start"]),
                    scene_end=int(scene_meta["scene_end"]),
                    anchor_start=int(snippet.get("start_char", 0)),
                    anchor_end=int(snippet.get("end_char", 0)),
                    max_context_chars=max_context_chars,
                )
                snippet["scene_ref"] = {
                    "scene_index": int(scene_meta["scene_index"]),
                    "scene_start": int(scene_meta["scene_start"]),
                    "scene_end": int(scene_meta["scene_end"]),
                    "max_context_chars": int(max_context_chars),
                }
                snippet["scene_excerpt"] = excerpt
            enriched_snippets.append(snippet)
        queue_items.append(
            ReviewQueueItem(
                queue_id=f"rq::{canonical_key}",
                canonical_key=canonical_key,
                task_id=result.task_id,
                source_document_paths=list(packet.source_document_paths),
                deterministic_prior={
                    "dominant_category": packet.payload.get("dominant_category", ""),
                    "source_keys": list(packet.payload.get("source_keys", [])),
                    "deterministic_reasons": list(packet.payload.get("reasons", [])),
                },
                first_pass_assessment={
                    "passing": bool(proposal.get("passing", False)),
                    "failing": bool(proposal.get("failing", proposal.get("review_required", False))),
                    "rationale_confidence": proposal.get("rationale_confidence"),
                    "review_required": bool(proposal.get("review_required", False)),
                    "uncertainty_reason": str(proposal.get("uncertainty_reason", "")),
                    "conflicting_categories": list(proposal.get("conflicting_categories", [])),
                    "proposed_category": str(
                        proposal.get("dominant_category", proposal.get("category", ""))
                    ),
                },
                evidence_snippets=enriched_snippets,
                max_context_chars=max_context_chars,
            )
        )

    queue_items.sort(key=lambda item: item.canonical_key)
    return queue_items
