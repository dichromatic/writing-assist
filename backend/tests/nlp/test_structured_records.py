"""Tests for the phase-1 structured-record dossier experiment."""

from pathlib import Path

from backend.nlp.experiments.dossier_review.cli import run_dossier_review_experiment
from backend.nlp.experiments.dossier_review.llm_pass import (
    _extract_json_object_text,
    run_dossier_llm_pass,
)
from backend.nlp.experiments.dossier_review.review_bundle import build_dossier_review_bundles
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.document_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.semantic_review import extract_reference_candidates
from backend.nlp.structured_records import build_dossier_seed_bundle, build_record_seed_bundle, segment_structured_records
from backend.nlp.types import StructuredFieldLineType, StructuredRecordType


def _document_outputs(path: str):
    """Run the current document pipeline on one source file.

    Args:
        path: Source document path.

    Returns:
        Parsed document plus document-level entity and reference hints.
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    doc = parse(path, raw_text)
    pre = preprocess(doc)
    result = bootstrap(doc)
    attribution_records = attribute_dialogue(pre, result.clusters)
    bundle = promote(pre, result.clusters, result.lexicon, attribution_records)
    entity_records = summarize_document_entities(pre, result.clusters, attribution_records, bundle)
    reference_candidates = extract_reference_candidates(pre, entity_records, attribution_records)
    return doc, entity_records, reference_candidates


def test_estuary_summary_segments_into_dossier_entries_with_banner_context():
    # The first experiment depends on stable record boundaries. If the banner
    # heading and subject headings are not separated correctly, every later
    # seed bundle will mix multiple crew entries together.
    path = "examples/story planning/estuary crew summaries.txt"
    doc = parse(path, Path(path).read_text(encoding="utf-8"))

    records = segment_structured_records(doc)
    dossier_records = [record for record in records if record.record_type == StructuredRecordType.DOSSIER_ENTRY]

    assert len(dossier_records) >= 7
    assert dossier_records[0].heading_text.startswith("🌊 WATANABE YŌ")
    assert dossier_records[0].parent_heading == "RADIANT ESTUARY — PRIMARY BRIDGE CREW SUMMARY"
    assert any(
        record.heading_text.startswith("IX. INTERPERSONAL RELATIONSHIPS")
        and record.record_type == StructuredRecordType.REFERENCE_SECTION
        for record in records
    )
    assert any(
        record.record_type == StructuredRecordType.DOSSIER_ENTRY
        and record.heading_text.startswith("Kurosawa Dia — Explorer")
        and record.parent_heading == "THE BRIDGE CREW OF THE RADIANT ESTUARY"
        for record in records
    )


def test_inline_character_synthesis_headers_become_dossier_entries_without_rank_row_noise():
    # The later synthesis block contains real per-character record boundaries,
    # while the preceding rank summary is only a compact roster. If the
    # segmenter cannot tell those apart, it either loses character records or
    # over-segments every one-line summary row into a fake dossier entry.
    path = "examples/story planning/estuary crew summaries.txt"
    doc = parse(path, Path(path).read_text(encoding="utf-8"))

    records = segment_structured_records(doc)
    dossier_headings = [record.heading_text for record in records if record.record_type == StructuredRecordType.DOSSIER_ENTRY]

    assert any(heading.startswith("Watanabe Yō — Pioneer-Admiral") for heading in dossier_headings)
    assert any(heading.startswith("Kurosawa Dia — Explorer") for heading in dossier_headings)
    assert not any(heading.startswith("Yō — Pioneer O‑9") for heading in dossier_headings)
    assert not any(heading.startswith("Mari — Pioneer O‑9") for heading in dossier_headings)


def test_dossier_seed_bundle_preserves_header_fields_and_subject_guess():
    # The deterministic seed bundle is the handoff contract. It must recover
    # shallow structure like the subject header, role field, and section-style
    # subheads without pretending to resolve the entry semantically.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    dossier_record = next(
        record for record in segment_structured_records(doc)
        if record.record_type == StructuredRecordType.DOSSIER_ENTRY
        and record.heading_text.startswith("🌊 WATANABE YŌ")
    )

    seed_bundle, subject_guess, fact_candidates = build_dossier_seed_bundle(
        dossier_record,
        entity_records,
        reference_candidates,
    )

    assert subject_guess is not None
    assert subject_guess.primary_guess == "Watanabe Yō"
    assert "CAPTAIN" in seed_bundle.candidate_rank_texts
    assert "PIONEER-ADMIRAL (O‑9)" in seed_bundle.candidate_rank_texts
    assert any(
        line.line_type == StructuredFieldLineType.LABEL_VALUE
        and line.label == "Role"
        for line in seed_bundle.field_lines
    )
    assert any(
        line.line_type == StructuredFieldLineType.STANDALONE_SUBHEAD
        and line.raw_text == "Vanguard History"
        for line in seed_bundle.field_lines
    )
    assert any(candidate.label == "Role" for candidate in fact_candidates)


def test_world_context_file_builds_reference_section_bundles():
    # World-context notes are the first non-dossier family that should become
    # proposal-bearing units. If they still collapse to diagnostics-only, the
    # shared structured-note path has not actually widened.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)

    bundles, diagnostics = build_dossier_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    assert bundles
    assert diagnostics.reason_no_dossier_bundles == ""
    assert diagnostics.candidate_record_counts["reference_section"] > 0
    assert any(bundle.record_type == StructuredRecordType.REFERENCE_SECTION for bundle in bundles)


def test_no_heading_world_context_file_still_segments_into_reference_sections():
    # Some reference notes use plain Roman-numbered lines instead of markdown
    # heading syntax. If the segmenter only trusts parsed heading spans, these
    # files collapse into one loose blob and the shared note path loses the
    # section units that later review depends on.
    path = "examples/world context/tau sectors.txt"
    doc = parse(path, Path(path).read_text(encoding="utf-8"))

    records = segment_structured_records(doc)

    assert any(record.record_type == StructuredRecordType.REFERENCE_SECTION for record in records)
    assert any(record.heading_text.startswith("I. The Universe Is Divided") for record in records)
    assert any(record.heading_text.startswith("II. What τ Means Physically") for record in records)


def test_reference_section_seed_bundle_preserves_heading_and_prose_statements():
    # Expository sections should carry their heading and explanatory prose as
    # explicit fact-like candidates, otherwise later normalization has no
    # structured handle for the core world-lore assertions.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    reference_record = next(
        record for record in segment_structured_records(doc)
        if record.record_type == StructuredRecordType.REFERENCE_SECTION
        and record.heading_text.startswith("### 1. Early Earth")
    )

    seed_bundle, subject_guess, fact_candidates = build_record_seed_bundle(
        reference_record,
        entity_records,
        reference_candidates,
    )

    assert subject_guess is None
    assert seed_bundle.header_line.startswith("### 1. Early Earth")
    assert any(candidate.label == "section_heading" for candidate in fact_candidates)
    assert any(candidate.label == "section_prose" for candidate in fact_candidates)
    assert all(candidate.value != "---" for candidate in fact_candidates)


def test_loose_record_seed_bundle_preserves_prelude_prose():
    # Heading-led reference files often have important framing prose before the
    # first section. Treating that prelude as diagnostic noise would lose
    # source-grounded context that later retrieval and LLM passes need.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    loose_record = next(
        record for record in segment_structured_records(doc)
        if record.record_type == StructuredRecordType.LOOSE_RECORD
    )

    seed_bundle, subject_guess, fact_candidates = build_record_seed_bundle(
        loose_record,
        entity_records,
        reference_candidates,
    )

    assert subject_guess is None
    assert seed_bundle.header_line == ""
    assert any(candidate.label == "loose_note" for candidate in fact_candidates)
    assert any("three long, partially overlapping arcs" in candidate.value for candidate in fact_candidates)


def test_review_bundle_includes_loose_records_as_supported_family():
    # Loose records are a real fallback family, not an error state. If the
    # bundle builder drops them, front matter and messy notes never reach the
    # same review path as more structured records.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)

    bundles, diagnostics = build_dossier_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    assert diagnostics.candidate_record_counts["loose_record"] > 0
    assert any(bundle.record_type == StructuredRecordType.LOOSE_RECORD for bundle in bundles)
    loose_bundle = next(bundle for bundle in bundles if bundle.record_type == StructuredRecordType.LOOSE_RECORD)
    assert loose_bundle.llm_prompt_packet.task_name == "loose_record_explicit_context"


def test_outline_beat_seed_bundle_preserves_heading_and_bullets():
    # Planning beats need their heading milestone and bullet steps preserved
    # explicitly, otherwise the shared note path still cannot carry planning
    # records forward into proposal normalization.
    path = "examples/story planning/Dia recovery arc.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    beat_record = next(
        record for record in segment_structured_records(doc)
        if record.record_type == StructuredRecordType.OUTLINE_BEAT
        and record.heading_text.startswith("0.1")
    )

    seed_bundle, subject_guess, fact_candidates = build_record_seed_bundle(
        beat_record,
        entity_records,
        reference_candidates,
    )

    assert subject_guess is None
    assert seed_bundle.header_line.startswith("0.1")
    assert any(candidate.label == "beat_heading" for candidate in fact_candidates)
    assert any(candidate.label == "beat_step" for candidate in fact_candidates)


def test_scene_and_arc_headings_do_not_segment_as_dossier_entries():
    # Planning scene headings and numbered arc legs can look superficially like
    # dashed subject headers. They are not character records, and treating them
    # as dossier entries makes the shared note path bleed structural planning
    # scaffolding into the wrong record family.
    briefing_path = "examples/story planning/briefing for mission chapter 10-11.txt"
    recovery_path = "examples/story planning/recovery arcs.txt"

    briefing_records = segment_structured_records(parse(
        briefing_path,
        Path(briefing_path).read_text(encoding="utf-8"),
    ))
    recovery_records = segment_structured_records(parse(
        recovery_path,
        Path(recovery_path).read_text(encoding="utf-8"),
    ))

    assert not any(
        record.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "SCENE 1a" in record.heading_text
        for record in briefing_records
    )
    assert not any(
        record.record_type == StructuredRecordType.DOSSIER_ENTRY
        and record.heading_text.startswith("5. KANAN + MARI")
        for record in recovery_records
    )


def test_experiment_writes_not_run_yet_placeholders(tmp_path):
    # The first experiment is deliberately pre-LLM. The persisted scaffold must
    # still expose explicit placeholder slots so the next phase can fill them
    # without redesigning the bundle shape.
    json_path, report_path, llm_report_path = run_dossier_review_experiment(
        "examples/story planning/prologue crew summaries.txt",
        str(tmp_path),
        max_report_records=2,
    )

    json_text = json_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    llm_report_text = llm_report_path.read_text(encoding="utf-8")

    assert '"status": "not_run_yet"' in json_text
    assert "llm_subject_proposal: not_run_yet" not in report_text
    assert "llm_fact_proposals: not_run_yet" not in report_text
    assert "llm_subject_proposal: not_run_yet" in llm_report_text
    assert "llm_fact_proposals: not_run_yet" in llm_report_text
    assert "STRUCTURED REVIEW BUNDLES" in report_text
    assert "🌊" not in json_text
    assert "☀️" not in json_text
    assert "🌊" not in report_text
    assert "☀️" not in report_text
    assert "🌊" not in llm_report_text
    assert "☀️" not in llm_report_text


def test_review_bundle_includes_explicit_llm_prompt_packet_with_weak_hints():
    # The next phase should plug a model into a frozen handoff contract, not
    # redesign the bundle. The prompt packet therefore needs to preserve the
    # narrow dossier task plus the full deterministic seed inventory,
    # including suppressed hints that manuscript-biased filtering would
    # otherwise hide.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)

    bundles, _diagnostics = build_dossier_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and bundle.deterministic_seed_bundle.header_line.startswith("🌊 WATANABE YŌ")
    )

    assert first_bundle.llm_prompt_packet.task_name == "dossier_subject_and_explicit_facts"
    assert first_bundle.llm_prompt_packet.source_authority == "planning_dossier"
    assert "🌊" not in first_bundle.llm_prompt_packet.header_line
    assert first_bundle.llm_prompt_packet.header_line.startswith("WATANABE YŌ")
    assert any(
        "explicit subject facts" in constraint.lower()
        or "explicit subject facts" in first_bundle.llm_prompt_packet.task_goal.lower()
        for constraint in first_bundle.llm_prompt_packet.task_constraints
    )
    assert (
        len(first_bundle.llm_prompt_packet.deterministic_seed_bundle.entity_candidates)
        == len(first_bundle.deterministic_seed_bundle.entity_candidates)
    )
    assert any(
        any(candidate.bucket.value == "suppressed" for candidate in bundle.llm_prompt_packet.deterministic_seed_bundle.entity_candidates)
        for bundle in bundles
    )


def test_llm_pass_fills_completed_payloads_and_comparison_fields():
    # The first live LLM step must enrich the existing bundle rather than
    # replacing it. This test locks that a completed response fills the
    # subject slot, fact slot, and side-by-side comparison fields from the
    # same deterministic baseline.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_dossier_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and bundle.deterministic_seed_bundle.header_line.startswith("🌊 WATANABE YŌ")
    )

    def fake_responder(_bundle, _model):
        return ({
            "subject": {
                "subject_name": "Watanabe Yō",
                "alternate_names": ["Yō"],
                "evidence_quotes": ["WATANABE YŌ — CAPTAIN / PIONEER-ADMIRAL (O‑9)"],
                "certainty_note": "Header line names the subject directly.",
                "unresolved": False,
            },
            "fact_proposals": [
                {
                    "label": "Role",
                    "value": "Commanding Officer, Radiant Estuary",
                    "evidence_quote": "Role: Commanding Officer, Radiant Estuary",
                    "certainty_note": "Explicit label-value line.",
                },
                {
                    "label": "header_rank",
                    "value": "CAPTAIN",
                    "evidence_quote": "WATANABE YŌ — CAPTAIN / PIONEER-ADMIRAL (O‑9)",
                    "certainty_note": "Explicit header rank text.",
                },
            ],
            "open_questions": ["Should 'Pioneer-Admiral (O‑9)' be stored as a second rank entry?"],
        }, "resp_test_123")

    updated_bundle = run_dossier_llm_pass(
        first_bundle,
        model="gpt-4o-mini",
        responder=fake_responder,
    )

    assert updated_bundle.llm_subject_proposal.status == "completed"
    assert updated_bundle.llm_subject_proposal.payload["subject_name"] == "Watanabe Yō"
    assert updated_bundle.llm_fact_proposals.status == "completed"
    assert updated_bundle.llm_fact_proposals.payload["items"][0]["label"] == "Role"
    assert "subject:Watanabe Yō" in updated_bundle.agreement_items
    assert "Role: Commanding Officer, Radiant Estuary" in updated_bundle.agreement_items
    assert "header_rank: PIONEER-ADMIRAL (O‑9)" in updated_bundle.deterministic_only_items
    assert updated_bundle.open_questions == [
        "Should 'Pioneer-Admiral (O‑9)' be stored as a second rank entry?"
    ]


def test_chat_completion_json_extractor_tolerates_fenced_output():
    # NVIDIA-style chat completions may wrap the JSON body in markdown fences
    # or prepend a short explanation. The live pass must still recover the
    # first structured object, otherwise failures only show up at runtime
    # against the provider.
    text = """
Here is the extracted JSON:

```json
{"subject":{"subject_name":"Watanabe Yō","alternate_names":[],"evidence_quotes":[],"certainty_note":"header","unresolved":false},"fact_proposals":[],"open_questions":[]}
```
"""

    assert _extract_json_object_text(text) == (
        '{"subject":{"subject_name":"Watanabe Yō","alternate_names":[],"evidence_quotes":[],"certainty_note":"header","unresolved":false},"fact_proposals":[],"open_questions":[]}'
    )


def test_deterministic_report_excludes_llm_details_but_llm_report_keeps_them(tmp_path):
    # The dossier scaffold now writes two text logs. The main report should stay
    # focused on deterministic record structure, while the separate LLM log
    # carries model status, comparisons, and review questions.
    json_path, report_path, llm_report_path = run_dossier_review_experiment(
        "examples/story planning/estuary crew summaries.txt",
        str(tmp_path),
        max_report_records=1,
        run_llm=True,
        max_llm_records=1,
    )

    _ = json_path
    report_text = report_path.read_text(encoding="utf-8")
    llm_report_text = llm_report_path.read_text(encoding="utf-8")

    assert "llm_subject_proposal:" not in report_text
    assert "llm_fact_proposals:" not in report_text
    assert "agreement_items:" not in report_text
    assert "open_questions:" not in report_text

    assert "llm_subject_proposal:" in llm_report_text
    assert "llm_fact_proposals:" in llm_report_text
    assert "open_questions:" in llm_report_text
