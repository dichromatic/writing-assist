"""Tests for database proposal projection and validation."""

from backend.nlp.indexing import (
    project_llm_task_results_to_database_proposals,
    project_task_packets_to_database_proposals,
    validate_database_proposals,
)
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


def _packet() -> LLMTaskPacket:
    """Build one minimal packet for projection tests."""
    return LLMTaskPacket(
        task_id="task-1",
        task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
        schema_id="record_fact_extraction.v1",
        source_bundle_kind="record_review_bundle",
        source_object_kind="structured_record",
        source_object_id="record-1",
        source_document_paths=["doc.txt"],
        document_type=DocumentType.WORLD_CONTEXT,
        document_status=DocumentStatus.PRIMARY_CANON,
        source_authority="structured_record:reference_section",
        source_authority_weight=1.0,
        task_goal="Extract record facts.",
        task_constraints=["No unsupported inference."],
        evidence_payload=[
            LLMTaskEvidenceItem(
                evidence_id="ev1",
                document_path="doc.txt",
                source_anchor=SpanAnchor(
                    path="doc.txt",
                    span_ordinal=0,
                    start_char=0,
                    end_char=12,
                ),
                quote="Role: Captain",
                context_before="",
                context_after="",
                source_object_id="record-1",
                visibility_bucket="deterministic_fact_candidate",
            )
        ],
        selection_reason="record_has_fact_candidates",
        payload={"record_type": "reference_section"},
    )


def test_task_packets_project_to_deterministic_database_proposals():
    # Handoff artifacts should be projectable into baseline deterministic
    # database proposals even before LLM results exist.
    proposals, diagnostics = project_task_packets_to_database_proposals([_packet()])

    assert len(proposals) == 1
    assert proposals[0].proposal_state.value == "deterministic_proposal"
    assert proposals[0].proposal_kind.value == "claim"
    assert diagnostics == []


def test_completed_llm_results_project_to_llm_database_proposals():
    # Completed task results should map into review-required LLM proposals with
    # source result ids preserved.
    packet = _packet()
    results = [
        LLMTaskResult(
            task_id="task-1",
            task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
            schema_id="record_fact_extraction.v1",
            status=LLMTaskResultStatus.COMPLETED,
            model="test-model",
            provider="test-provider",
            response_id="resp-1",
            payload={"items": [{"label": "Role", "value": "Captain"}]},
        )
    ]
    proposals, diagnostics = project_llm_task_results_to_database_proposals(results, [packet])

    assert len(proposals) == 1
    assert proposals[0].proposal_state.value == "llm_proposal"
    assert proposals[0].source_result_ids == ["resp-1"]
    assert diagnostics == []


def test_validation_sets_insertability_states():
    # Validation must separate insertable deterministic proposals from LLM
    # proposals that still need normalization.
    deterministic, _ = project_task_packets_to_database_proposals([_packet()])
    llm, _ = project_llm_task_results_to_database_proposals(
        [
            LLMTaskResult(
                task_id="task-1",
                task_family=LLMTaskFamily.RECORD_FACT_EXTRACTION,
                schema_id="record_fact_extraction.v1",
                status=LLMTaskResultStatus.COMPLETED,
                model="test-model",
                provider="test-provider",
                response_id="resp-2",
                payload={"items": []},
            )
        ],
        [_packet()],
    )

    validated, diagnostics = validate_database_proposals(deterministic + llm)
    by_state = {item.proposal_state.value: item.insertability_state.value for item in validated}

    assert by_state["deterministic_proposal"] == "insertable"
    assert by_state["llm_proposal"] == "needs_normalization"
    assert any(item.code == "llm_observed_not_normalized" for item in diagnostics)
