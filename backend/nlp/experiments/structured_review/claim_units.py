"""
Structured review claim units - project review bundles into retrieval-shaped claims.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle list] --> B[Iterate deterministic fact candidates]
        B --> C[Build local claim groups]
        C --> D[Build ClaimEvidence]
        D --> E[Build ClaimUnit]
        E --> F[Attach same-group neighbor ids]
        F --> G[ClaimUnit list]
"""

from __future__ import annotations

from typing import Any

from backend.nlp.harvesting.shared import stable_hash_id
from backend.nlp.types import (
    ClaimEvidence,
    ClaimGroup,
    ClaimKind,
    ClaimProposalState,
    ClaimReviewState,
    ClaimUnit,
    DeterministicFactCandidate,
    RecordReviewBundle,
    SpanAnchor,
    StructuredRecordType,
)

_FIELD_BUNDLE_LABELS = {
    "header_rank",
    "section_heading",
    "beat_heading",
    "loose_label",
}
_BULLET_LABELS = {
    "section_bullet",
    "beat_step",
    "loose_bullet",
}
_PROSE_LABELS = {
    "section_prose",
    "beat_note",
    "loose_note",
}
_ALIAS_LABELS = {
    "alias",
    "aliases",
    "alternate name",
    "alternate names",
    "aka",
}
_RELATION_LABEL_HINTS = {
    "relationship",
    "relationships",
    "relation",
    "dynamic",
    "dynamics",
    "bond",
    "family",
}
_EVENT_LABEL_HINTS = {
    "history",
    "vanguard history",
    "outcome",
    "result",
    "narrative purpose",
}


def _claim_kind(label: str, record_type: StructuredRecordType) -> ClaimKind:
    """Return a conservative claim kind for one deterministic candidate.

    Args:
        label: Candidate label.
        record_type: Source structured record family.

    Returns:
        Coarse claim kind for retrieval grouping.
    """
    normalized = label.strip().casefold()
    if normalized in _ALIAS_LABELS:
        return ClaimKind.ALIAS
    if normalized in _RELATION_LABEL_HINTS or "relationship" in normalized:
        return ClaimKind.RELATION
    if record_type == StructuredRecordType.OUTLINE_BEAT:
        return ClaimKind.EVENT
    if normalized in _EVENT_LABEL_HINTS:
        return ClaimKind.EVENT
    return ClaimKind.FACT


def _group_kind(label: str) -> str:
    """Return the local claim-group kind for one deterministic candidate.

    Args:
        label: Candidate label.

    Returns:
        Group kind string from the retrieval planning note's small set.
    """
    normalized = label.strip().casefold()
    if normalized in _BULLET_LABELS:
        return "bullet_bundle"
    if normalized in _PROSE_LABELS:
        return "prose_bundle"
    if normalized in _FIELD_BUNDLE_LABELS:
        return "field_bundle"
    if normalized in _RELATION_LABEL_HINTS or "relationship" in normalized:
        return "relation_bundle"
    return "field_bundle"


def _group_label(candidate: DeterministicFactCandidate) -> str:
    """Return a display label for a claim group.

    Args:
        candidate: Deterministic candidate being projected.

    Returns:
        Stable display label for local grouping.
    """
    if candidate.line_index >= 0:
        return f"{candidate.label}:{candidate.line_index}"
    return candidate.label


def _subject_guess(bundle: RecordReviewBundle) -> tuple[str, list[str]]:
    """Return the primary and alternate subject guesses for a bundle.

    Args:
        bundle: Source review bundle.

    Returns:
        Primary subject guess plus alternate candidates.
    """
    if bundle.deterministic_subject_guess is None:
        return "", []
    return (
        bundle.deterministic_subject_guess.primary_guess,
        list(bundle.deterministic_subject_guess.alternative_guesses),
    )


def _llm_subject_guess(bundle: RecordReviewBundle) -> tuple[str, list[str]]:
    """Return the completed LLM subject guess for a bundle.

    Args:
        bundle: Source review bundle.

    Returns:
        Primary subject guess plus alternate candidates.
    """
    if bundle.llm_subject_proposal.status != "completed":
        return "", []
    payload = bundle.llm_subject_proposal.payload
    if payload.get("unresolved", False):
        return "", [
            str(candidate)
            for candidate in payload.get("alternate_names", [])
            if str(candidate).strip()
        ]
    subject_name = str(payload.get("subject_name", "")).strip()
    alternate_names = [
        str(candidate).strip()
        for candidate in payload.get("alternate_names", [])
        if str(candidate).strip()
    ]
    return subject_name, alternate_names


def _claim_summary(subject: str, label: str, value: str) -> str:
    """Return a readable claim summary.

    Args:
        subject: Optional subject string.
        label: Claim label.
        value: Claim value.

    Returns:
        Human-readable one-line summary.
    """
    if subject:
        return f"{subject} - {label}: {value}"
    return f"{label}: {value}"


def _structure_quality(bundle: RecordReviewBundle) -> float:
    """Return a simple structure-quality score for a claim source.

    Args:
        bundle: Source review bundle.

    Returns:
        Score in the range [0.0, 1.0].
    """
    if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY:
        return 0.9
    if bundle.record_type == StructuredRecordType.REFERENCE_SECTION:
        return 0.75
    if bundle.record_type == StructuredRecordType.OUTLINE_BEAT:
        return 0.7
    if bundle.record_type == StructuredRecordType.LOOSE_RECORD:
        return 0.45
    return 0.0


def _claim_group(
    bundle: RecordReviewBundle,
    candidate: DeterministicFactCandidate,
) -> ClaimGroup:
    """Build a local claim group for one deterministic candidate.

    Args:
        bundle: Source review bundle.
        candidate: Deterministic candidate being projected.

    Returns:
        Local claim group metadata.
    """
    group_label = _group_label(candidate)
    group_kind = _group_kind(candidate.label)
    group_id = stable_hash_id(
        bundle.record_id,
        "claim_group",
        group_kind,
        group_label,
    )
    evidence_id = stable_hash_id(
        bundle.record_id,
        "evidence",
        candidate.label,
        str(candidate.line_index),
        candidate.value,
    )
    return ClaimGroup(
        claim_group_id=group_id,
        source_record_id=bundle.record_id,
        group_kind=group_kind,
        group_label=group_label,
        primary_evidence_id=evidence_id,
    )


def _llm_claim_group(
    bundle: RecordReviewBundle,
    fact_item: dict[str, Any],
    item_index: int,
) -> ClaimGroup:
    """Build a local claim group for one LLM fact proposal.

    Args:
        bundle: Source review bundle.
        fact_item: LLM fact proposal payload.
        item_index: Stable index within the model fact list.

    Returns:
        Local claim group metadata.
    """
    label = str(fact_item.get("label", "")).strip()
    group_kind = _group_kind(label)
    group_label = f"llm:{label}:{item_index}"
    group_id = stable_hash_id(
        bundle.record_id,
        "llm_claim_group",
        group_kind,
        group_label,
    )
    evidence_id = stable_hash_id(
        bundle.record_id,
        "llm_evidence",
        label,
        str(item_index),
        str(fact_item.get("evidence_quote", "")),
    )
    return ClaimGroup(
        claim_group_id=group_id,
        source_record_id=bundle.record_id,
        group_kind=group_kind,
        group_label=group_label,
        primary_evidence_id=evidence_id,
    )


def _bundle_fallback_anchor(bundle: RecordReviewBundle) -> SpanAnchor:
    """Return the best available source anchor for model-derived claims.

    Args:
        bundle: Source review bundle.

    Returns:
        Span anchor from deterministic evidence already attached to the bundle.

    Raises:
        ValueError: If the bundle has no available source anchor.
    """
    if bundle.deterministic_fact_candidates:
        return bundle.deterministic_fact_candidates[0].supporting_anchor
    if bundle.deterministic_subject_guess is not None:
        return bundle.deterministic_subject_guess.supporting_anchor
    if bundle.deterministic_seed_bundle.entity_candidates:
        return bundle.deterministic_seed_bundle.entity_candidates[0].anchors[0]
    if bundle.deterministic_seed_bundle.reference_candidates:
        return bundle.deterministic_seed_bundle.reference_candidates[0].anchor
    raise ValueError(f"Review bundle has no source anchor: {bundle.record_id}")


def _claim_from_candidate(
    bundle: RecordReviewBundle,
    candidate: DeterministicFactCandidate,
) -> ClaimUnit:
    """Project one deterministic fact candidate into a claim unit.

    Args:
        bundle: Source review bundle.
        candidate: Deterministic candidate.

    Returns:
        Retrieval-shaped claim unit.
    """
    primary_subject, alternate_subjects = _subject_guess(bundle)
    group = _claim_group(bundle, candidate)
    claim_id = stable_hash_id(
        bundle.record_id,
        "claim_unit",
        candidate.label,
        str(candidate.line_index),
        candidate.value,
    )
    evidence = ClaimEvidence(
        anchor=candidate.supporting_anchor,
        quote=candidate.value,
        source_snippet=bundle.raw_text,
        evidence_role="primary",
    )
    claim_value = candidate.value.strip()
    claim_label = candidate.label.strip()
    return ClaimUnit(
        claim_id=claim_id,
        claim_kind=_claim_kind(candidate.label, bundle.record_type),
        primary_subject_guess=primary_subject,
        alternate_subject_candidates=alternate_subjects,
        claim_label=claim_label,
        claim_value=claim_value,
        readable_summary=_claim_summary(primary_subject, claim_label, claim_value),
        raw_claim_payload={
            "candidate_label": candidate.label,
            "candidate_value": candidate.value,
            "candidate_reason": candidate.reason,
            "line_index": candidate.line_index,
        },
        source_record_id=bundle.record_id,
        source_document_path=bundle.document_path,
        document_type=bundle.document_type,
        source_family=bundle.record_type.value,
        source_status=bundle.document_status,
        source_authority=bundle.llm_prompt_packet.source_authority,
        source_authority_weight=bundle.llm_prompt_packet.source_authority_weight,
        primary_evidence=evidence,
        supporting_evidence=[],
        retrieval_channel_tags=["literal_local"],
        retrieval_reasons=["source_record_neighbor"],
        primary_retrieval_reason="source_record_neighbor",
        review_state=ClaimReviewState.UNREVIEWED,
        proposal_state=ClaimProposalState.DETERMINISTIC_PROPOSAL,
        claim_group=group,
        structure_quality=_structure_quality(bundle),
    )


def _claim_from_llm_fact(
    bundle: RecordReviewBundle,
    fact_item: dict[str, Any],
    item_index: int,
) -> ClaimUnit:
    """Project one completed LLM fact proposal into a claim unit.

    Args:
        bundle: Source review bundle.
        fact_item: LLM fact proposal payload.
        item_index: Stable index within the model fact list.

    Returns:
        Retrieval-shaped claim unit.
    """
    primary_subject, alternate_subjects = _llm_subject_guess(bundle)
    claim_label = str(fact_item.get("label", "")).strip()
    claim_value = str(fact_item.get("value", "")).strip()
    evidence_quote = str(fact_item.get("evidence_quote", "")).strip() or claim_value
    certainty_note = str(fact_item.get("certainty_note", "")).strip()
    group = _llm_claim_group(bundle, fact_item, item_index)
    claim_id = stable_hash_id(
        bundle.record_id,
        "llm_claim_unit",
        str(item_index),
        claim_label,
        claim_value,
        evidence_quote,
    )
    evidence = ClaimEvidence(
        anchor=_bundle_fallback_anchor(bundle),
        quote=evidence_quote,
        source_snippet=bundle.raw_text,
        evidence_role="primary",
    )
    return ClaimUnit(
        claim_id=claim_id,
        claim_kind=_claim_kind(claim_label, bundle.record_type),
        primary_subject_guess=primary_subject,
        alternate_subject_candidates=alternate_subjects,
        claim_label=claim_label,
        claim_value=claim_value,
        readable_summary=_claim_summary(primary_subject, claim_label, claim_value),
        raw_claim_payload={
            "llm_label": claim_label,
            "llm_value": claim_value,
            "evidence_quote": evidence_quote,
            "certainty_note": certainty_note,
            "model": bundle.llm_fact_proposals.model,
            "response_id": bundle.llm_fact_proposals.response_id,
        },
        source_record_id=bundle.record_id,
        source_document_path=bundle.document_path,
        document_type=bundle.document_type,
        source_family=bundle.record_type.value,
        source_status=bundle.document_status,
        source_authority=bundle.llm_prompt_packet.source_authority,
        source_authority_weight=bundle.llm_prompt_packet.source_authority_weight,
        primary_evidence=evidence,
        supporting_evidence=[],
        retrieval_channel_tags=["semantic_inferred"],
        retrieval_reasons=["inferred_subject_match"],
        primary_retrieval_reason="inferred_subject_match",
        review_state=ClaimReviewState.REVIEW_REQUIRED,
        proposal_state=ClaimProposalState.LLM_PROPOSAL,
        claim_group=group,
        structure_quality=_structure_quality(bundle),
    )


def build_claim_units_from_review_bundles(
    bundles: list[RecordReviewBundle],
) -> list[ClaimUnit]:
    """Build retrieval-shaped claim units from review bundles.

    Args:
        bundles: Structured review bundles.

    Returns:
        Claim units in stable source order.
    """
    claim_units: list[ClaimUnit] = []
    for bundle in bundles:
        for candidate in bundle.deterministic_fact_candidates:
            if not candidate.value.strip():
                continue
            claim_units.append(_claim_from_candidate(bundle, candidate))
        if bundle.llm_fact_proposals.status != "completed":
            continue
        for item_index, fact_item in enumerate(bundle.llm_fact_proposals.payload.get("items", [])):
            claim_label = str(fact_item.get("label", "")).strip()
            claim_value = str(fact_item.get("value", "")).strip()
            if not claim_label or not claim_value:
                continue
            claim_units.append(_claim_from_llm_fact(bundle, fact_item, item_index))

    group_to_claim_ids: dict[str, list[str]] = {}
    for claim_unit in claim_units:
        if claim_unit.claim_group is None:
            continue
        group_to_claim_ids.setdefault(
            claim_unit.claim_group.claim_group_id,
            [],
        ).append(claim_unit.claim_id)

    for claim_unit in claim_units:
        if claim_unit.claim_group is None:
            continue
        claim_unit.neighbor_claim_ids = [
            claim_id
            for claim_id in group_to_claim_ids[claim_unit.claim_group.claim_group_id]
            if claim_id != claim_unit.claim_id
        ]
    return claim_units
