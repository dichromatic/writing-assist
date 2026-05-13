"""
Database proposal projection - convert deterministic and LLM outputs into shared proposals.

.. code-block:: mermaid

    flowchart TD
        A[RecordReviewBundle list] --> B[Deterministic claim projection]
        C[ManuscriptReviewBundle] --> D[Entity and review question projection]
        E[LLMTaskResult list] --> F[Task-family proposal projection]
        B & D & F --> G[DatabaseProposal list]
        B & D & F --> H[IndexingDiagnostic list]
"""

from __future__ import annotations

from typing import Any

from backend.nlp.types import (
    DatabaseProposal,
    DatabaseProposalApprovalState,
    DatabaseProposalEvidenceRef,
    DatabaseProposalInsertabilityState,
    DatabaseProposalKind,
    DatabaseProposalReviewState,
    DatabaseProposalState,
    DocumentStatus,
    DocumentType,
    IndexingDiagnostic,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskResult,
    LLMTaskResultStatus,
    ManuscriptReviewBundle,
    RecordReviewBundle,
    stable_hash_id,
)


def _quality_annotations_for_result(
    *,
    packet: LLMTaskPacket,
    proposal_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return non-blocking quality annotations for one LLM proposal."""
    annotations: list[dict[str, Any]] = []

    if packet.task_family == LLMTaskFamily.RECORD_FACT_EXTRACTION:
        facts = proposal_payload.get("facts", [])
        if isinstance(facts, list):
            heading_like = [
                item for item in facts
                if isinstance(item, dict)
                and str(item.get("statement", "")).strip().lower().startswith("section heading")
            ]
            if heading_like:
                annotations.append(
                    {
                        "code": "quality_heading_metadata_fact",
                        "message": "Detected heading-style metadata emitted as lore fact.",
                        "count": len(heading_like),
                    }
                )
            if packet.payload.get("record_type") == "loose_record":
                raw_text = str(packet.payload.get("raw_record_text", ""))
                parroted = [
                    item for item in facts
                    if isinstance(item, dict)
                    and str(item.get("statement", "")).strip()
                    and str(item.get("statement", "")).strip() in raw_text
                    and len(str(item.get("statement", "")).strip()) >= 120
                ]
                if parroted:
                    annotations.append(
                        {
                            "code": "quality_parroted_source",
                            "message": "Detected likely verbatim long-form source parroting.",
                            "count": len(parroted),
                        }
                    )

    if packet.task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
        evidence = packet.evidence_payload
        if evidence:
            thin = sum(
                1
                for item in evidence
                if not item.context_before.strip() and not item.context_after.strip()
            )
            if thin == len(evidence):
                annotations.append(
                    {
                        "code": "quality_context_too_thin",
                        "message": "All supporting evidence items have empty context windows.",
                        "count": thin,
                    }
                )
        review_required = bool(proposal_payload.get("review_required", False))
        uncertainty_reason = str(proposal_payload.get("uncertainty_reason", "")).strip()
        category = str(
            proposal_payload.get("dominant_category", proposal_payload.get("category", ""))
        ).strip().lower()
        if review_required:
            if uncertainty_reason:
                annotations.append(
                    {
                        "code": "quality_deferral_justified",
                        "message": "Deferral includes explicit uncertainty rationale.",
                    }
                )
            else:
                annotations.append(
                    {
                        "code": "quality_deferral_unjustified",
                        "message": "Deferral lacks explicit uncertainty rationale.",
                    }
                )
            if category in {"character", "place", "group", "organization"} and not uncertainty_reason:
                annotations.append(
                    {
                        "code": "quality_over_deferral_risk",
                        "message": "Deferred despite resolved category without rationale.",
                    }
                )
    return annotations


def _canonicalize_manuscript_review_semantics(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return manuscript payload with stable review-semantic keys."""
    normalized = dict(payload)
    review_required = normalized.get("review_required")
    if not isinstance(review_required, bool):
        review_required = False

    uncertainty_reason = normalized.get("uncertainty_reason")
    if not isinstance(uncertainty_reason, str):
        uncertainty_reason = ""
    uncertainty_reason = uncertainty_reason.strip()

    conflicting = normalized.get("conflicting_categories")
    if not isinstance(conflicting, list):
        conflicting = []
    conflicting_categories = [str(item).strip() for item in conflicting if str(item).strip()]

    normalized["review_required"] = review_required
    normalized["uncertainty_reason"] = uncertainty_reason
    normalized["conflicting_categories"] = conflicting_categories
    return normalized


def _result_payload_and_validation(
    result_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, list[str], dict[str, Any]]:
    """Extract normalized proposal payload and validation metadata."""
    if "proposal_payload" in result_payload:
        return (
            dict(result_payload.get("proposal_payload", {})),
            bool(result_payload.get("is_valid", False)),
            list(result_payload.get("validation_errors", [])),
            dict(result_payload.get("raw_payload", {})),
        )
    return dict(result_payload), True, [], dict(result_payload)


def _base_proposal(
    *,
    proposal_id: str,
    proposal_kind: DatabaseProposalKind,
    proposal_state: DatabaseProposalState,
    source_bundle_kind: str,
    source_object_kind: str,
    source_object_id: str,
    source_document_paths: list[str],
    document_type: DocumentType,
    document_status: DocumentStatus,
    source_authority: str,
    source_authority_weight: float,
    evidence_refs: list[DatabaseProposalEvidenceRef],
    evidence_quotes: list[str],
    subject_keys: list[str],
    retrieval_tags: list[str],
    payload: dict[str, Any],
    raw_source_payload: dict[str, Any],
    source_result_ids: list[str] | None = None,
) -> DatabaseProposal:
    """Build one database proposal with default workflow and insertability states."""
    return DatabaseProposal(
        proposal_id=proposal_id,
        proposal_kind=proposal_kind,
        proposal_state=proposal_state,
        review_state=DatabaseProposalReviewState.UNREVIEWED,
        approval_state=DatabaseProposalApprovalState.NOT_REVIEWED,
        insertability_state=DatabaseProposalInsertabilityState.NEEDS_NORMALIZATION,
        source_bundle_kind=source_bundle_kind,
        source_object_kind=source_object_kind,
        source_object_id=source_object_id,
        source_document_paths=source_document_paths,
        document_type=document_type,
        document_status=document_status,
        source_authority=source_authority,
        source_authority_weight=source_authority_weight,
        evidence_refs=evidence_refs,
        evidence_quotes=evidence_quotes,
        subject_keys=subject_keys,
        retrieval_tags=retrieval_tags,
        payload=payload,
        raw_source_payload=raw_source_payload,
        source_result_ids=list(source_result_ids or []),
    )


def project_record_review_bundles_to_database_proposals(
    bundles: list[RecordReviewBundle],
) -> tuple[list[DatabaseProposal], list[IndexingDiagnostic]]:
    """Project deterministic structured-record review bundles into claim proposals."""
    proposals: list[DatabaseProposal] = []
    diagnostics: list[IndexingDiagnostic] = []
    for bundle in bundles:
        subject = (
            bundle.deterministic_subject_guess.primary_guess
            if bundle.deterministic_subject_guess is not None
            else ""
        )
        for candidate in bundle.deterministic_fact_candidates:
            if not candidate.value.strip():
                diagnostics.append(
                    IndexingDiagnostic(
                        code="deterministic_observed_not_normalized",
                        level="warning",
                        source_bundle_kind="record_review_bundle",
                        source_object_kind="structured_record",
                        source_object_id=bundle.record_id,
                        message="Skipped empty deterministic fact candidate.",
                        context={"label": candidate.label},
                    )
                )
                continue
            evidence_ref = DatabaseProposalEvidenceRef(
                evidence_id=stable_hash_id(
                    "proposal_evidence",
                    bundle.record_id,
                    candidate.label,
                    str(candidate.line_index),
                ),
                document_path=bundle.document_path,
                anchor=candidate.supporting_anchor,
                evidence_role="primary",
            )
            proposal = _base_proposal(
                proposal_id=stable_hash_id(
                    "database_proposal",
                    "record_claim",
                    bundle.record_id,
                    candidate.label,
                    candidate.value,
                    str(candidate.line_index),
                ),
                proposal_kind=DatabaseProposalKind.CLAIM,
                proposal_state=DatabaseProposalState.DETERMINISTIC_PROPOSAL,
                source_bundle_kind="record_review_bundle",
                source_object_kind="structured_record",
                source_object_id=bundle.record_id,
                source_document_paths=[bundle.document_path],
                document_type=bundle.document_type,
                document_status=bundle.document_status,
                source_authority=f"structured_record:{bundle.record_type.value}",
                source_authority_weight=1.0,
                evidence_refs=[evidence_ref],
                evidence_quotes=[candidate.value],
                subject_keys=[subject] if subject else [],
                retrieval_tags=["literal_local", bundle.record_type.value],
                payload={
                    "claim_label": candidate.label,
                    "claim_value": candidate.value,
                    "candidate_reason": candidate.reason,
                    "line_index": candidate.line_index,
                },
                raw_source_payload={
                    "record_type": bundle.record_type.value,
                    "header_line": bundle.deterministic_seed_bundle.header_line,
                },
            )
            proposal.review_state = DatabaseProposalReviewState.REVIEW_REQUIRED
            proposals.append(proposal)
    return proposals, diagnostics


def project_manuscript_review_bundle_to_database_proposals(
    bundle: ManuscriptReviewBundle,
) -> tuple[list[DatabaseProposal], list[IndexingDiagnostic]]:
    """Project manuscript deterministic outputs into database proposals."""
    proposals: list[DatabaseProposal] = []
    diagnostics: list[IndexingDiagnostic] = []
    for entity in bundle.canonical_entities:
        if not entity.member_records:
            diagnostics.append(
                IndexingDiagnostic(
                    code="deterministic_observed_not_normalized",
                    level="warning",
                    source_bundle_kind="manuscript_review_bundle",
                    source_object_kind="corpus_entity",
                    source_object_id=entity.canonical_key,
                    message="Skipped entity_profile proposal due to empty member records.",
                )
            )
            continue
        record = entity.member_records[0]
        if not record.anchors:
            continue
        evidence_ref = DatabaseProposalEvidenceRef(
            evidence_id=stable_hash_id(
                "proposal_evidence",
                "manuscript_entity_profile",
                entity.canonical_key,
            ),
            document_path=record.document_anchor.path,
            anchor=record.anchors[0],
            evidence_role="primary",
        )
        proposal = _base_proposal(
            proposal_id=stable_hash_id(
                "database_proposal",
                "entity_profile",
                entity.canonical_key,
            ),
            proposal_kind=DatabaseProposalKind.ENTITY_PROFILE,
            proposal_state=DatabaseProposalState.DETERMINISTIC_PROPOSAL,
            source_bundle_kind="manuscript_review_bundle",
            source_object_kind="corpus_entity",
            source_object_id=entity.canonical_key,
            source_document_paths=list(entity.supporting_document_paths),
            document_type=DocumentType.MANUSCRIPT,
            document_status=DocumentStatus.PRIMARY_CANON,
            source_authority="manuscript_corpus",
            source_authority_weight=1.0,
            evidence_refs=[evidence_ref],
            evidence_quotes=[record.surface_forms[0] if record.surface_forms else record.normalized_key],
            subject_keys=[entity.canonical_key],
            retrieval_tags=["entity_profile", entity.dominant_category.value],
            payload={
                "dominant_category": entity.dominant_category.value,
                "source_keys": list(entity.source_keys),
                "review_required": entity.review_required,
                "reasons": list(entity.reasons),
            },
            raw_source_payload={},
        )
        proposal.review_state = (
            DatabaseProposalReviewState.REVIEW_REQUIRED
            if entity.review_required
            else DatabaseProposalReviewState.UNREVIEWED
        )
        proposals.append(proposal)

    for task in bundle.review_tasks:
        proposal = _base_proposal(
            proposal_id=stable_hash_id(
                "database_proposal",
                "open_review_question",
                task.task_id,
            ),
            proposal_kind=DatabaseProposalKind.OPEN_REVIEW_QUESTION,
            proposal_state=DatabaseProposalState.DETERMINISTIC_PROPOSAL,
            source_bundle_kind="manuscript_review_bundle",
            source_object_kind="review_task",
            source_object_id=task.task_id,
            source_document_paths=list(task.supporting_anchor_paths),
            document_type=DocumentType.MANUSCRIPT,
            document_status=DocumentStatus.PRIMARY_CANON,
            source_authority="manuscript_corpus",
            source_authority_weight=1.0,
            evidence_refs=[],
            evidence_quotes=[],
            subject_keys=[task.subject_key] if task.subject_key else [],
            retrieval_tags=["open_review_question", task.kind.value],
            payload={
                "prompt": task.prompt,
                "kind": task.kind.value,
                "ranked_candidate_keys": list(task.ranked_candidate_keys),
                "ranked_speaker_keys": list(task.ranked_speaker_keys),
                "corpus_owner_keys": list(task.corpus_owner_keys),
                "evidence_note": task.evidence_note,
            },
            raw_source_payload={},
        )
        proposal.review_state = DatabaseProposalReviewState.REVIEW_REQUIRED
        proposals.append(proposal)

    return proposals, diagnostics


def _kind_from_task_family(task_family: LLMTaskFamily) -> DatabaseProposalKind:
    """Map one LLM task family to a proposal kind."""
    mapping = {
        LLMTaskFamily.RECORD_FACT_EXTRACTION: DatabaseProposalKind.CLAIM,
        LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE: DatabaseProposalKind.ENTITY_PROFILE,
        LLMTaskFamily.MANUSCRIPT_REFERENCE_ATTACHMENT: DatabaseProposalKind.REFERENCE_ATTACHMENT,
        LLMTaskFamily.MANUSCRIPT_CATEGORY_RESOLUTION: DatabaseProposalKind.CATEGORY_RESOLUTION,
        LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION: DatabaseProposalKind.ENTITY_PROFILE,
    }
    return mapping[task_family]


def project_task_packets_to_database_proposals(
    task_packets: list[LLMTaskPacket],
) -> tuple[list[DatabaseProposal], list[IndexingDiagnostic]]:
    """Project task packets into deterministic staging proposals.

    This provides a uniform baseline when only handoff artifacts are available.
    """
    proposals: list[DatabaseProposal] = []
    diagnostics: list[IndexingDiagnostic] = []
    for packet in task_packets:
        evidence_refs = [
            DatabaseProposalEvidenceRef(
                evidence_id=item.evidence_id,
                document_path=item.document_path,
                anchor=item.source_anchor,
                evidence_role="primary",
            )
            for item in packet.evidence_payload
        ]
        proposal = _base_proposal(
            proposal_id=stable_hash_id(
                "database_proposal",
                "task_packet_baseline",
                packet.task_id,
            ),
            proposal_kind=_kind_from_task_family(packet.task_family),
            proposal_state=DatabaseProposalState.DETERMINISTIC_PROPOSAL,
            source_bundle_kind=packet.source_bundle_kind,
            source_object_kind=packet.source_object_kind,
            source_object_id=packet.source_object_id,
            source_document_paths=list(packet.source_document_paths),
            document_type=packet.document_type,
            document_status=packet.document_status,
            source_authority=packet.source_authority,
            source_authority_weight=packet.source_authority_weight,
            evidence_refs=evidence_refs,
            evidence_quotes=[item.quote for item in packet.evidence_payload[:8]],
            subject_keys=[],
            retrieval_tags=[packet.task_family.value, "task_packet_baseline"],
            payload={"task_goal": packet.task_goal, "payload": dict(packet.payload)},
            raw_source_payload={},
        )
        proposal.review_state = DatabaseProposalReviewState.REVIEW_REQUIRED
        proposals.append(proposal)
    return proposals, diagnostics


def project_llm_task_results_to_database_proposals(
    results: list[LLMTaskResult],
    task_packets: list[LLMTaskPacket],
) -> tuple[list[DatabaseProposal], list[IndexingDiagnostic]]:
    """Project completed LLM task results into shared LLM proposal objects."""
    proposals: list[DatabaseProposal] = []
    diagnostics: list[IndexingDiagnostic] = []
    packet_by_id = {packet.task_id: packet for packet in task_packets}
    first_pass_entity_proposal_id_by_key: dict[str, str] = {}
    for result in results:
        packet = packet_by_id.get(result.task_id)
        if packet is None:
            diagnostics.append(
                IndexingDiagnostic(
                    code="llm_observed_not_normalized",
                    level="warning",
                    source_bundle_kind="unknown",
                    source_object_kind="llm_task_result",
                    source_object_id=result.task_id,
                    message="Missing task packet for LLM task result.",
                )
            )
            continue
        if result.status != LLMTaskResultStatus.COMPLETED:
            diagnostics.append(
                IndexingDiagnostic(
                    code="llm_observed_not_normalized",
                    level="info",
                    source_bundle_kind=packet.source_bundle_kind,
                    source_object_kind=packet.source_object_kind,
                    source_object_id=packet.source_object_id,
                    message=f"Skipped non-completed result with status={result.status.value}.",
                )
            )
            continue
        proposal_payload, is_valid, validation_errors, raw_payload = _result_payload_and_validation(
            result.payload
        )
        if result.task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
            proposal_payload = _canonicalize_manuscript_review_semantics(proposal_payload)
        evidence_refs = [
            DatabaseProposalEvidenceRef(
                evidence_id=item.evidence_id,
                document_path=item.document_path,
                anchor=item.source_anchor,
                evidence_role="primary",
            )
            for item in packet.evidence_payload
        ]
        proposal = _base_proposal(
            proposal_id=stable_hash_id(
                "database_proposal",
                "llm_task_result",
                result.task_id,
                result.response_id or "",
            ),
            proposal_kind=_kind_from_task_family(result.task_family),
            proposal_state=DatabaseProposalState.LLM_PROPOSAL,
            source_bundle_kind=packet.source_bundle_kind,
            source_object_kind=packet.source_object_kind,
            source_object_id=packet.source_object_id,
            source_document_paths=list(packet.source_document_paths),
            document_type=packet.document_type,
            document_status=packet.document_status,
            source_authority=packet.source_authority,
            source_authority_weight=packet.source_authority_weight,
            evidence_refs=evidence_refs,
            evidence_quotes=[item.quote for item in packet.evidence_payload[:8]],
            subject_keys=[],
            retrieval_tags=[
                result.task_family.value,
                "llm_inferred",
                "llm_validated" if is_valid else "llm_validation_failed",
            ],
            payload=proposal_payload,
            raw_source_payload={
                "response_id": result.response_id,
                "provider": result.provider,
                "model": result.model,
                "task_goal": packet.task_goal,
                "llm_validation": {
                    "is_valid": is_valid,
                    "validation_errors": validation_errors,
                },
                "llm_raw_payload": raw_payload,
            },
            source_result_ids=[result.response_id] if result.response_id else [],
        )
        if result.task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE:
            canonical = str(
                proposal_payload.get("canonical_key")
                or proposal_payload.get("entity_key")
                or proposal.source_object_id
            )
            if canonical:
                first_pass_entity_proposal_id_by_key[canonical] = proposal.proposal_id
        if result.task_family == LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION:
            proposal.retrieval_tags.append("review_resolution")
            resolved = bool(proposal_payload.get("resolved", False))
            if resolved:
                proposal.payload["review_resolution_priority"] = "prefer_over_first_pass"
            canonical = str(
                proposal_payload.get("canonical_key")
                or proposal.source_object_id
            )
            parent_id = first_pass_entity_proposal_id_by_key.get(canonical)
            if parent_id:
                proposal.parent_proposal_ids.append(parent_id)
        quality_annotations = _quality_annotations_for_result(
            packet=packet,
            proposal_payload=proposal_payload,
        )
        if quality_annotations:
            proposal.raw_source_payload["quality_annotations"] = quality_annotations
            for annotation in quality_annotations:
                diagnostics.append(
                    IndexingDiagnostic(
                        code=str(annotation.get("code", "quality_annotation")),
                        level="info",
                        source_bundle_kind=packet.source_bundle_kind,
                        source_object_kind=packet.source_object_kind,
                        source_object_id=packet.source_object_id,
                        message=str(annotation.get("message", "LLM quality annotation.")),
                        context={"proposal_id": proposal.proposal_id, **annotation},
                    )
                )
        proposal.review_state = DatabaseProposalReviewState.REVIEW_REQUIRED
        proposals.append(proposal)
    return proposals, diagnostics
