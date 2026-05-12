"""
Database proposal validation - determine insertability and emit diagnostics.

.. code-block:: mermaid

    flowchart TD
        A[DatabaseProposal list] --> B[Validate envelope fields]
        B --> C[Validate payload and evidence]
        C --> D[Set insertability state]
        D --> E[Validated proposals]
        C --> F[IndexingDiagnostic list]
"""

from __future__ import annotations

from backend.nlp.types import (
    DatabaseProposal,
    DatabaseProposalInsertabilityState,
    DatabaseProposalKind,
    IndexingDiagnostic,
)

def _llm_payload_valid_for_insertion(proposal: DatabaseProposal) -> bool:
    """Return True when the LLM proposal payload passed typed normalization."""
    details = proposal.raw_source_payload.get("llm_validation", {})
    if not isinstance(details, dict):
        return False
    return bool(details.get("is_valid", False))


def _missing_required_fields(proposal: DatabaseProposal) -> list[str]:
    """Return missing required proposal envelope fields."""
    missing: list[str] = []
    if not proposal.proposal_id:
        missing.append("proposal_id")
    if not proposal.source_bundle_kind:
        missing.append("source_bundle_kind")
    if not proposal.source_object_kind:
        missing.append("source_object_kind")
    if not proposal.source_object_id:
        missing.append("source_object_id")
    if not proposal.source_document_paths:
        missing.append("source_document_paths")
    if proposal.proposal_kind in {
        DatabaseProposalKind.CLAIM,
        DatabaseProposalKind.ENTITY_PROFILE,
        DatabaseProposalKind.REFERENCE_ATTACHMENT,
        DatabaseProposalKind.CATEGORY_RESOLUTION,
    }:
        if not proposal.evidence_refs:
            missing.append("evidence_refs")
    return missing


def validate_database_proposals(
    proposals: list[DatabaseProposal],
) -> tuple[list[DatabaseProposal], list[IndexingDiagnostic]]:
    """Validate proposals and set insertability state with diagnostics."""
    diagnostics: list[IndexingDiagnostic] = []
    for proposal in proposals:
        missing = _missing_required_fields(proposal)
        proposal.validation_errors = []
        if missing:
            proposal.validation_errors.extend(f"missing:{field}" for field in missing)
            proposal.insertability_state = DatabaseProposalInsertabilityState.NOT_INSERTABLE
            diagnostics.append(
                IndexingDiagnostic(
                    code="proposal_not_insertable",
                    level="error",
                    source_bundle_kind=proposal.source_bundle_kind,
                    source_object_kind=proposal.source_object_kind,
                    source_object_id=proposal.source_object_id,
                    message="Proposal missing required envelope fields.",
                    context={"missing_fields": missing, "proposal_id": proposal.proposal_id},
                )
            )
            continue
        if proposal.proposal_state.value == "llm_proposal":
            if _llm_payload_valid_for_insertion(proposal):
                proposal.insertability_state = DatabaseProposalInsertabilityState.INSERTABLE
            else:
                proposal.insertability_state = DatabaseProposalInsertabilityState.NEEDS_NORMALIZATION
                diagnostics.append(
                    IndexingDiagnostic(
                        code="llm_observed_not_normalized",
                        level="info",
                        source_bundle_kind=proposal.source_bundle_kind,
                        source_object_kind=proposal.source_object_kind,
                        source_object_id=proposal.source_object_id,
                        message="LLM proposal requires normalization before insertion.",
                        context={"proposal_id": proposal.proposal_id},
                    )
                )
            continue
        proposal.insertability_state = DatabaseProposalInsertabilityState.INSERTABLE
    return proposals, diagnostics
