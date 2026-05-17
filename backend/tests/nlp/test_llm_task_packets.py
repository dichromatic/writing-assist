"""Tests for shared LLM task-packet generation from deterministic review bundles."""

from pathlib import Path

from backend.nlp.experiments.structured_review.cli import run_structured_review_experiment
from backend.nlp.experiments.structured_review.review_bundle import build_structured_review_bundles
from backend.nlp.llm_tasks.assembly.builders import build_llm_task_packets
from backend.nlp.llm_tasks.assembly.structured_records import (
    build_structured_record_tagged_extraction_task_packets,
    build_structured_record_task_packets,
)
from backend.nlp.parsing.document_parser import parse
from backend.nlp.structured_records import extract_structural_entities, segment_structured_records
from backend.nlp.types import LLMTaskFamily


def test_structured_record_task_packets_are_built_from_deterministic_review_bundles():
    # The handoff contract now requires deterministic review bundles to project
    # into shared task packets without provider execution.
    path = "examples/world context/human history.txt"
    raw = Path(path).read_text(encoding="utf-8")
    records = segment_structured_records(parse(path, raw))
    entity_inventory = extract_structural_entities(records)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_inventory=entity_inventory,
    )

    packets, selection_diagnostics = build_structured_record_task_packets(bundles)

    assert packets
    assert selection_diagnostics
    assert all(packet.task_family == LLMTaskFamily.RECORD_FACT_EXTRACTION for packet in packets)
    assert all(packet.schema_id == "record_fact_extraction.v1" for packet in packets)
    assert any(item.selected for item in selection_diagnostics)
    assert any(item.reason == "record_has_fact_candidates" for item in selection_diagnostics)


def test_structured_record_tagged_extraction_packets_are_built():
    # The new tagged extraction family should build one packet per selected
    # record with schema-stable metadata and structural payload context.
    path = "examples/world context/human history.txt"
    raw = Path(path).read_text(encoding="utf-8")
    records = segment_structured_records(parse(path, raw))
    entity_inventory = extract_structural_entities(records)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_inventory=entity_inventory,
    )

    packets, selection_diagnostics = build_structured_record_tagged_extraction_task_packets(bundles)

    assert packets
    assert selection_diagnostics
    assert all(packet.task_family == LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION for packet in packets)
    assert all(packet.schema_id == "structured_record_tagged_extraction.v1" for packet in packets)
    assert any(item.selected for item in selection_diagnostics)
    assert any(item.reason == "record_has_structural_context" for item in selection_diagnostics)


def test_builder_can_include_structured_tagged_extraction_packets():
    # Shared packet builder should optionally include the new structured tagged
    # extraction family alongside existing record-fact extraction packets.
    path = "examples/world context/human history.txt"
    raw = Path(path).read_text(encoding="utf-8")
    records = segment_structured_records(parse(path, raw))
    entity_inventory = extract_structural_entities(records)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_inventory=entity_inventory,
    )

    packets, diagnostics = build_llm_task_packets(
        record_review_bundles=bundles,
        include_structured_tagged_extraction=True,
    )

    families = {packet.task_family for packet in packets}
    assert LLMTaskFamily.RECORD_FACT_EXTRACTION in families
    assert LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION in families
    assert any(item.task_family == LLMTaskFamily.STRUCTURED_RECORD_TAGGED_EXTRACTION for item in diagnostics)


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
    assert '"structured_record_tagged_extraction"' in payload
    assert llm_report_path.name.endswith("-structured-review-llm-task-packets.txt")
