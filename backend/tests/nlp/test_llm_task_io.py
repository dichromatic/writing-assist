"""Tests for LLM task artifact IO helpers."""

from backend.nlp.llm_tasks.io import extract_rescued_keys_from_results
from backend.nlp.types import LLMTaskFamily, LLMTaskResult, LLMTaskResultStatus


def _result(
    *,
    status: LLMTaskResultStatus = LLMTaskResultStatus.COMPLETED,
    payload: dict | None = None,
) -> LLMTaskResult:
    """Build one minimal LLM task result for rescue-key extraction tests."""
    return LLMTaskResult(
        task_id="t1",
        task_family=LLMTaskFamily.MANUSCRIPT_SUPPRESSION_RESCUE,
        schema_id="manuscript_suppression_rescue.v1",
        status=status,
        model="m",
        provider="p",
        payload=payload or {},
    )


def test_extract_rescued_keys_from_results_keeps_only_valid_rescue_true_keys():
    """Only completed, valid, rescue-true payloads should emit keys."""
    results = [
        _result(payload={
            "is_valid": True,
            "proposal_payload": {"normalized_key": "firth", "rescue": True},
        }),
        _result(payload={
            "is_valid": True,
            "proposal_payload": {"normalized_key": "aurora", "rescue": False},
        }),
        _result(payload={
            "is_valid": False,
            "proposal_payload": {"normalized_key": "mari", "rescue": True},
        }),
        _result(
            status=LLMTaskResultStatus.FAILED,
            payload={
                "is_valid": True,
                "proposal_payload": {"normalized_key": "star", "rescue": True},
            },
        ),
        _result(payload={
            "is_valid": True,
            "proposal_payload": {"normalized_key": " ", "rescue": True},
        }),
    ]

    assert extract_rescued_keys_from_results(results) == frozenset({"firth"})
