"""Tests for shared LLM task-packet generation from deterministic review bundles."""

from pathlib import Path

from backend.nlp.experiments.structured_review.cli import run_structured_review_experiment
from backend.nlp.experiments.structured_review.review_bundle import build_structured_review_bundles
from backend.nlp.llm_tasks.assembly.structured_records import (
    build_structured_record_task_packets,
)
from backend.nlp.pipeline import run_document_pipeline
from backend.nlp.parsing.document_parser import parse
from backend.nlp.structured_records import segment_structured_records
from backend.nlp.types import LLMTaskFamily


def test_structured_record_task_packets_are_built_from_deterministic_review_bundles():
    # The handoff contract now requires deterministic review bundles to project
    # into shared task packets without provider execution.
    path = "examples/world context/human history.txt"
    raw = Path(path).read_text(encoding="utf-8")
    pipeline = run_document_pipeline(path, raw)
    records = segment_structured_records(parse(path, raw))
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        pipeline.entity_records,
        pipeline.reference_candidates,
    )

    packets, selection_diagnostics = build_structured_record_task_packets(bundles)

    assert packets
    assert selection_diagnostics
    assert all(packet.task_family == LLMTaskFamily.RECORD_FACT_EXTRACTION for packet in packets)
    assert all(packet.schema_id == "record_fact_extraction.v1" for packet in packets)
    assert any(item.selected for item in selection_diagnostics)
    assert any(item.reason == "record_has_fact_candidates" for item in selection_diagnostics)


def test_structured_review_cli_writes_task_packet_artifact_fields(tmp_path):
    # The structured-review CLI should emit shared task packets and selection
    # diagnostics in the JSON artifact, even before provider execution exists.
    input_path = "examples/world context/human history.txt"
    json_path, _report_path, llm_report_path = run_structured_review_experiment(
        input_path,
        str(tmp_path),
        max_report_records=3,
    )

    payload = json_path.read_text(encoding="utf-8")

    assert '"llm_task_packets"' in payload
    assert '"llm_task_diagnostics"' in payload
    assert llm_report_path.name.endswith("-structured-review-llm-task-packets.txt")
