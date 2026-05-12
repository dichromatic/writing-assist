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

from backend.nlp.document_metadata import document_status_authority_weight
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
        source_authority=f"structured_record:{bundle.record_type.value}",
        source_authority_weight=document_status_authority_weight(bundle.document_status),
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
