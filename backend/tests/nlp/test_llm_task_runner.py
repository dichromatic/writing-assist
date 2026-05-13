"""Tests for shared LLM task runner and artifact IO."""

import json
from pathlib import Path

from backend.nlp.llm_tasks.execution.io import load_task_packets_from_artifact
from backend.nlp.llm_tasks.execution.models import normalize_llm_payload
from backend.nlp.llm_tasks.execution.provider import (
    _extract_json_object_text,
    run_llm_task_packets,
)
from backend.nlp.types import (
    DocumentStatus,
    DocumentType,
    LLMTaskEvidenceItem,
    LLMTaskFamily,
    LLMTaskPacket,
    LLMTaskResultStatus,
    SpanAnchor,
)


def _packet() -> LLMTaskPacket:
    """Build one minimal shared task packet for runner tests."""
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
        task_goal="Extract evidence-backed facts.",
        task_constraints=["No unsupported inference."],
        evidence_payload=[
            LLMTaskEvidenceItem(
                evidence_id="evidence-1",
                document_path="doc.txt",
                source_anchor=SpanAnchor(
                    path="doc.txt",
                    span_ordinal=0,
                    start_char=0,
                    end_char=10,
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


def test_runner_marks_packets_skipped_when_no_responder():
    # The shared runner must remain safe by default. Without a configured
    # responder, packets are skipped rather than accidentally sent anywhere.
    results = run_llm_task_packets(
        [_packet()],
        model="dry_model",
        provider="dry_run",
        responder=None,
    )
    assert len(results) == 1
    assert results[0].status == LLMTaskResultStatus.SKIPPED
    assert results[0].error == "no responder configured"


def test_load_task_packets_from_handoff_artifact(tmp_path: Path):
    # Task runner input should be independent from source family and load from
    # shared handoff artifacts without bespoke parser logic per artifact type.
    packet = _packet()
    artifact = {
        "review_bundle_artifact_version": "1",
        "source_kind": "structured_record",
        "review_bundle_kind": "record_review_bundle_list",
        "review_bundle": [],
        "llm_task_packets": [
            {
                "task_id": packet.task_id,
                "task_family": packet.task_family.value,
                "schema_id": packet.schema_id,
                "source_bundle_kind": packet.source_bundle_kind,
                "source_object_kind": packet.source_object_kind,
                "source_object_id": packet.source_object_id,
                "source_document_paths": packet.source_document_paths,
                "document_type": packet.document_type.value,
                "document_status": packet.document_status.value,
                "source_authority": packet.source_authority,
                "source_authority_weight": packet.source_authority_weight,
                "task_goal": packet.task_goal,
                "task_constraints": packet.task_constraints,
                "selection_reason": packet.selection_reason,
                "payload": packet.payload,
                "evidence_payload": [
                    {
                        "evidence_id": packet.evidence_payload[0].evidence_id,
                        "document_path": packet.evidence_payload[0].document_path,
                        "source_anchor": {
                            "path": packet.evidence_payload[0].source_anchor.path,
                            "span_ordinal": packet.evidence_payload[0].source_anchor.span_ordinal,
                            "start_char": packet.evidence_payload[0].source_anchor.start_char,
                            "end_char": packet.evidence_payload[0].source_anchor.end_char,
                        },
                        "quote": packet.evidence_payload[0].quote,
                        "context_before": "",
                        "context_after": "",
                        "source_object_id": packet.evidence_payload[0].source_object_id,
                        "visibility_bucket": packet.evidence_payload[0].visibility_bucket,
                    }
                ],
            }
        ],
        "llm_task_diagnostics": [],
    }
    artifact_path = tmp_path / "handoff.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_task_packets_from_artifact(str(artifact_path))

    assert len(loaded) == 1
    assert loaded[0].task_id == "task-1"
    assert loaded[0].task_family == LLMTaskFamily.RECORD_FACT_EXTRACTION
    assert loaded[0].document_type == DocumentType.WORLD_CONTEXT


def test_json_object_extractor_handles_fenced_model_content():
    # Provider responses can wrap JSON in markdown fences. The extractor must
    # recover the object body before payload decoding.
    text = """
```json
{"items":[{"label":"Role","value":"Captain"}]}
```
"""
    assert _extract_json_object_text(text) == '{"items":[{"label":"Role","value":"Captain"}]}'


def test_review_resolution_unresolved_forces_review_required_true():
    # Second-pass unresolved results must stay open for review regardless of
    # model-provided review_required value.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        raw_payload={
            "canonical_key": "admiral",
            "resolved": False,
            "review_required": False,
            "remaining_uncertainty": "insufficient context",
        },
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["resolved"] is False
    assert envelope.proposal_payload["review_required"] is True


def test_review_resolution_sets_resolution_candidate_when_strong_but_unresolved():
    # Unresolved responses with explicit category plus rationale and no
    # remaining uncertainty should be marked as resolution candidates.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        raw_payload={
            "canonical_key": "admiralty",
            "resolved": False,
            "resolved_category": "organization",
            "review_required": True,
            "resolution_notes": "Usage consistently indicates institutional authority.",
            "remaining_uncertainty": "",
        },
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["resolution_candidate"] is True
    assert "institutional authority" in envelope.proposal_payload["candidate_reason"]


def test_review_resolution_does_not_set_candidate_when_uncertainty_present():
    # When uncertainty remains explicit, unresolved outputs should stay
    # non-candidate and continue through review.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        raw_payload={
            "canonical_key": "admiral",
            "resolved": False,
            "resolved_category": "character",
            "review_required": True,
            "resolution_rationale": "Some supporting narrative context exists.",
            "remaining_uncertainty": "Multiple distinct admirals remain possible.",
        },
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["resolution_candidate"] is False


def test_pass1_semantics_derive_failing_and_rationale_confidence():
    # Pass-1 triage responses may use legacy confidence naming. The normalizer
    # should preserve review semantics under the new passing/failing contract.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
        raw_payload={
            "canonical_key": "admiral",
            "dominant_category": "character",
            "review_required": True,
            "confidence": 0.72,
        },
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["failing"] is True
    assert envelope.proposal_payload["passing"] is False
    assert envelope.proposal_payload["rationale_confidence"] == 0.72


def test_pass1_empty_payload_is_rejected_by_completeness_guard():
    # Completed pass-1 results with empty proposal payload must not be treated
    # as valid because triage routing depends on explicit flags.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
        raw_payload={},
    )
    assert envelope.is_valid is False
    assert any("missing required triage booleans" in item for item in envelope.validation_errors)


def test_pass1_confidence_label_is_coerced_to_numeric():
    # Some model variants emit confidence labels. Coercion should preserve
    # strict schema validation while keeping outputs usable.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_PROFILE,
        raw_payload={
            "canonical_key": "dia",
            "dominant_category": "character",
            "passing": True,
            "failing": False,
            "review_required": False,
            "rationale_confidence": "high",
        },
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["rationale_confidence"] == 0.85


def test_pass2_empty_payload_is_rejected_by_completeness_guard():
    # Completed pass-2 results with empty proposal payload must be rejected so
    # unresolved entities do not appear silently resolved.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        raw_payload={},
    )
    assert envelope.is_valid is False
    assert any("missing required resolution key" in item for item in envelope.validation_errors)


def test_pass2_backfills_missing_canonical_key_from_packet_fallback():
    # Missing canonical key should be recoverable from deterministic packet
    # identity during normalization.
    envelope = normalize_llm_payload(
        task_family=LLMTaskFamily.MANUSCRIPT_ENTITY_REVIEW_RESOLUTION,
        raw_payload={
            "resolved": False,
            "passing": False,
            "failing": True,
            "review_required": True,
            "remaining_uncertainty": "ambiguous",
        },
        fallback_canonical_key="admiralty",
    )
    assert envelope.is_valid is True
    assert envelope.proposal_payload["canonical_key"] == "admiralty"
