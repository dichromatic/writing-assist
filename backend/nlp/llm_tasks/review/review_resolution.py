"""
Review-resolution task builder - deferred manuscript entities to second-pass packets.

.. code-block:: mermaid

    flowchart TD
        A[First-pass results] --> B[Filter manuscript_entity_profile review_required]
        C[First-pass packets] --> D[Index by task_id]
        B & D --> E[Build ReviewQueueItem list]
        E --> F[Build manuscript_entity_review_resolution packets]
"""

from __future__ import annotations

from backend.nlp.llm_tasks.assembly.schemas import schema_id_for
from backend.nlp.llm_tasks.review.review_queue import build_manuscript_review_queue
from backend.nlp.types import (
    DocumentStatus,
    DocumentType,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskSelectionDiagnostic,
    SpanAnchor,
    stable_hash_id,
)


def build_review_resolution_task_packets(
    *,
    first_pass_packets: list[LLMTaskPacket],
    first_pass_results: list[LLMTaskResult],
    max_snippets: int = 5,
    max_context_chars: int = 2000,
    max_tasks: int | None = None,
) -> tuple[list[LLMTaskPacket], list[LLMTaskSelectionDiagnostic]]:
    """Build second-pass review-resolution packets from first-pass deferred entities."""
    queue_items = build_manuscript_review_queue(
        task_packets=first_pass_packets,
        task_results=first_pass_results,
        max_snippets=max_snippets,
        max_context_chars=max_context_chars,
    )
    if max_tasks is not None:
        queue_items = queue_items[: max(0, max_tasks)]

    packets: list[LLMTaskPacket] = []
    diagnostics: list[LLMTaskSelectionDiagnostic] = []
    for item in queue_items:
        evidence_items: list[LLMTaskEvidenceItem] = []
        for snippet in item.evidence_snippets:
            evidence_items.append(
                LLMTaskEvidenceItem(
                    evidence_id=str(snippet.get("evidence_id", "")),
                    document_path=str(snippet.get("document_path", "")),
                    source_anchor=SpanAnchor(
                        path=str(snippet.get("document_path", "")),
                        span_ordinal=int(snippet.get("span_ordinal", 0)),
                        start_char=int(snippet.get("start_char", 0)),
                        end_char=int(snippet.get("end_char", 0)),
                    ),
                    quote=str(snippet.get("quote", "")),
                    context_before=str(snippet.get("context_before", "")),
                    context_after=str(snippet.get("context_after", "")),
                    source_object_id=item.canonical_key,
                    visibility_bucket=str(snippet.get("visibility_bucket", "review_only")),
                    confidence_score=float(snippet.get("confidence_score", 0.0)),
                    evidence_metadata={
                        "scene_ref": dict(snippet.get("scene_ref", {})),
                        "scene_excerpt": str(snippet.get("scene_excerpt", "")),
                    },
                )
            )

        packets.append(
            LLMTaskPacket(
                task_id=stable_hash_id(
                    "llm_task_packet",
                    LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION.value,
                    item.canonical_key,
                ),
                task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
                schema_id=schema_id_for(LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION),
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="corpus_entity",
                source_object_id=item.canonical_key,
                source_document_paths=list(item.source_document_paths),
                document_type=DocumentType.MANUSCRIPT,
                document_status=DocumentStatus.PRIMARY_CANON,
                source_authority="manuscript_corpus",
                source_authority_weight=1.0,
                task_goal="Resolve deferred manuscript entity using richer context and first-pass uncertainty rationale.",
                task_constraints=[
                    "Use scene-context evidence plus first-pass rationale.",
                    "Resolve only when evidence is explicit.",
                    "If unresolved, keep review_required true with remaining_uncertainty.",
                ],
                evidence_payload=evidence_items,
                selection_reason="first_pass_review_required",
                payload={
                    "queue_id": item.queue_id,
                    "first_pass_task_id": item.task_id,
                    "deterministic_prior": item.deterministic_prior,
                    "first_pass_assessment": item.first_pass_assessment,
                    "max_context_chars": item.max_context_chars,
                },
            )
        )
        diagnostics.append(
            LLMTaskSelectionDiagnostic(
                source_bundle_kind="manuscript_review_bundle",
                source_object_kind="corpus_entity",
                source_object_id=item.canonical_key,
                document_path=item.source_document_paths[0] if item.source_document_paths else "",
                task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
                selected=True,
                reason="first_pass_review_required",
                evidence_counts={"snippets": len(item.evidence_snippets)},
            )
        )
    return packets, diagnostics
