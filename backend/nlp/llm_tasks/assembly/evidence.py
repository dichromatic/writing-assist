# Diagram omitted - utility module with no significant information flow.

"""
Shared evidence-item construction for LLM task packet assembly.

Both manuscript and structured-record task builders emit the same
``LLMTaskEvidenceItem`` shape. The only difference is whether the caller has
real surrounding context to provide. This helper keeps the constructor logic
in one place while leaving manuscript and structured-record evidence
selection rules separate.
"""

from __future__ import annotations

from backend.nlp.types import LLMTaskEvidenceItem, SpanAnchor, stable_hash_id


def build_evidence_item(
    *,
    source_object_id: str,
    anchor: SpanAnchor,
    quote: str,
    visibility_bucket: str,
    context_before: str = "",
    context_after: str = "",
    suppression_reason: str = "",
    confidence_score: float | None = None,
) -> LLMTaskEvidenceItem:
    """Build one bounded evidence item for an LLM task packet.

    Args:
        source_object_id: Stable identifier of the source object that owns this
            evidence item.
        anchor: Precise source anchor for the evidence span.
        quote: Short quoted surface form or fact text for the evidence item.
        visibility_bucket: Evidence lane or source bucket label.
        context_before: Optional surrounding left context.
        context_after: Optional surrounding right context.
        suppression_reason: Optional suppression rationale when the evidence
            comes from a weaker lane.
        confidence_score: Optional deterministic confidence to preserve.

    Returns:
        A stable ``LLMTaskEvidenceItem`` ready for packet assembly.
    """
    return LLMTaskEvidenceItem(
        evidence_id=stable_hash_id(
            "llm_task_evidence",
            source_object_id,
            anchor.path,
            str(anchor.start_char),
            str(anchor.end_char),
            quote,
        ),
        document_path=anchor.path,
        source_anchor=anchor,
        quote=quote,
        context_before=context_before,
        context_after=context_after,
        source_object_id=source_object_id,
        visibility_bucket=visibility_bucket,
        suppression_reason=suppression_reason,
        confidence_score=confidence_score,
    )
