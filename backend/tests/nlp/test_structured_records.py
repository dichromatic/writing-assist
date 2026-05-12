"""Tests for the structured-record review experiment."""

from pathlib import Path

from backend.nlp.document_metadata import (
    document_status_authority_weight,
    resolve_document_metadata,
)
from backend.nlp.document_type import classify_document_type
from backend.nlp.experiments.structured_review.claim_units import build_claim_units_from_review_bundles
from backend.nlp.experiments.structured_review.cli import run_structured_review_experiment
from backend.nlp.experiments.structured_review.llm_pass import (
    _extract_json_object_text,
    run_structured_llm_pass,
)
from backend.nlp.experiments.structured_review.report import render_structured_llm_report
from backend.nlp.experiments.structured_review.review_bundle import build_structured_review_bundles
from backend.nlp.lexicon.bootstrap import bootstrap
from backend.nlp.parsing.document_parser import parse
from backend.nlp.parsing.preprocessing import preprocess
from backend.nlp.promotion.attribution import attribute_dialogue
from backend.nlp.promotion.promotion import promote
from backend.nlp.reconciliation.document_entities import summarize_document_entities
from backend.nlp.semantic_review import extract_reference_candidates
from backend.nlp.structured_records import build_record_seed_bundle, segment_structured_records
from backend.nlp.types import (
    ClaimKind,
    DocumentStatus,
    DocumentType,
    StructuredFieldLineType,
    StructuredRecordType,
)


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
    promotion_result = promote(pre, result.clusters, result.lexicon, attribution_records)
    entity_records = summarize_document_entities(
        pre,
        result.clusters,
        promotion_result.bundle,
        promotion_result.scores,
        promotion_result.classifications,
    )
    reference_candidates = extract_reference_candidates(pre, entity_records, attribution_records)
    return doc, entity_records, reference_candidates


def test_document_type_classifier_preserves_corpus_file_classification():
    # Document type is a corpus-file classification, not a record-family
    # segmentation result. These examples lock that distinction before the
    # value is used in persistence and retrieval.
    assert classify_document_type("examples/story planning/recovery arcs.txt") == DocumentType.STORY_PLANNING
    assert classify_document_type("examples/world context/human history.txt") == DocumentType.WORLD_CONTEXT
    assert classify_document_type("examples/vignettes/camilla rinka 3.md") == DocumentType.VIGNETTE
    assert classify_document_type("examples/locations/Solar System.docx") == DocumentType.LOCATION
    assert (
        classify_document_type("examples/character backgrounds/Nilam Norre.docx")
        == DocumentType.CHARACTER_BACKGROUND
    )
    assert classify_document_type("examples/4. Tairngire.md") == DocumentType.MANUSCRIPT


def test_document_metadata_defaults_to_primary_canon_without_explicit_status():
    # Default status needs to be explicit in artifacts. Otherwise later
    # persistence cannot tell whether authority metadata was forgotten or
    # intentionally treated as primary canon.
    metadata = resolve_document_metadata(
        "examples/world context/human history.txt",
        "Human history notes",
    )

    assert metadata.document_type == DocumentType.WORLD_CONTEXT
    assert metadata.document_status == DocumentStatus.PRIMARY_CANON
    assert metadata.status_source == "default"
    assert metadata.metadata_conflicts == []


def test_document_metadata_accepts_matching_in_document_and_sidecar_status():
    # Sidecar manifests and in-document headers should be able to agree without
    # making the document reviewable. This keeps future batch metadata usable
    # while still allowing author-visible per-file status.
    metadata = resolve_document_metadata(
        "examples/vignettes/camilla rinka 3.md",
        "document_status: legendary\n\nA remembered account.",
        {
            "documents": {
                "examples/vignettes/camilla rinka 3.md": {"status": "legendary"}
            }
        },
    )

    assert metadata.document_type == DocumentType.VIGNETTE
    assert metadata.document_status == DocumentStatus.LEGENDARY
    assert metadata.status_source == "sidecar:examples/vignettes/camilla rinka 3.md"
    assert metadata.status_hints == ["folder:vignettes"]
    assert metadata.metadata_conflicts == []


def test_document_metadata_conflict_downgrades_to_draft_unknown():
    # Conflicting metadata must not silently pick one authority source. The
    # document remains ingestible, but downstream review sees the conflict and
    # a conservative status.
    metadata = resolve_document_metadata(
        "examples/story planning/recovery arcs.txt",
        "document_status: historical\n\nArc notes.",
        {"recovery arcs.txt": "primary_canon"},
    )

    assert metadata.document_status == DocumentStatus.DRAFT_UNKNOWN
    assert metadata.status_source == "conflict"
    assert metadata.metadata_conflicts == [
        "document_status mismatch: in_document=historical sidecar=primary_canon"
    ]


def test_folder_status_hints_do_not_override_default_or_explicit_status():
    # Vignette-like folders are useful hints, but the user decided status is
    # per-document. Folder names must not silently convert the source away from
    # primary canon when no explicit status exists.
    default_metadata = resolve_document_metadata(
        "examples/vignettes/camilla rinka 3.md",
        "A remembered account.",
    )
    explicit_metadata = resolve_document_metadata(
        "examples/vignettes/camilla rinka 3.md",
        'document_status: "historical"\n\nA remembered account.',
    )

    assert default_metadata.document_status == DocumentStatus.PRIMARY_CANON
    assert default_metadata.status_hints == ["folder:vignettes"]
    assert explicit_metadata.document_status == DocumentStatus.HISTORICAL
    assert explicit_metadata.status_hints == ["folder:vignettes"]


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

    seed_bundle, subject_guess, fact_candidates = build_record_seed_bundle(
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

    bundles, diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    assert bundles
    assert diagnostics.document_type == DocumentType.WORLD_CONTEXT
    assert diagnostics.document_status == DocumentStatus.PRIMARY_CANON
    assert diagnostics.document_status_source == "default"
    assert diagnostics.reason_no_review_bundles == ""
    assert diagnostics.candidate_record_counts["reference_section"] > 0
    assert any(bundle.record_type == StructuredRecordType.REFERENCE_SECTION for bundle in bundles)
    assert all(bundle.document_type == DocumentType.WORLD_CONTEXT for bundle in bundles)
    assert all(bundle.document_status == DocumentStatus.PRIMARY_CANON for bundle in bundles)


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

    bundles, diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    assert diagnostics.candidate_record_counts["loose_record"] > 0
    assert any(bundle.record_type == StructuredRecordType.LOOSE_RECORD for bundle in bundles)
    loose_bundle = next(bundle for bundle in bundles if bundle.record_type == StructuredRecordType.LOOSE_RECORD)
    assert loose_bundle.llm_prompt_packet.task_name == "loose_record_explicit_context"


def test_claim_units_project_deterministic_facts_as_atomic_retrieval_units():
    # The retrieval handoff needs one small claim per deterministic fact
    # candidate. Bundling an entire record into one claim would make ranking,
    # citation, and later review too coarse.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    claim_units = [
        claim_unit
        for claim_unit in build_claim_units_from_review_bundles([first_bundle])
        if claim_unit.claim_label == "Role"
    ]

    assert len(claim_units) == 1
    claim_unit = claim_units[0]
    assert claim_unit.claim_kind == ClaimKind.FACT
    assert claim_unit.primary_subject_guess == "Watanabe Yō"
    assert claim_unit.claim_value == "Commanding Officer, Radiant Estuary"
    assert claim_unit.document_type == DocumentType.STORY_PLANNING
    assert claim_unit.source_record_id == first_bundle.record_id
    assert claim_unit.source_family == "dossier_entry"
    assert claim_unit.source_status == DocumentStatus.PRIMARY_CANON
    assert claim_unit.source_authority == "planning_dossier"
    assert claim_unit.source_authority_weight == document_status_authority_weight(DocumentStatus.PRIMARY_CANON)
    assert claim_unit.proposal_state.value == "deterministic_proposal"
    assert claim_unit.review_state.value == "unreviewed"
    assert claim_unit.primary_evidence.quote == "Commanding Officer, Radiant Estuary"


def test_claim_units_group_same_source_line_neighbors():
    # Claim groups are local to a source record and preserve authored grouping
    # without forcing multiple facts into one blob. Header rank claims share a
    # field bundle and should point at each other as neighbors.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    rank_claims = [
        claim_unit
        for claim_unit in build_claim_units_from_review_bundles([first_bundle])
        if claim_unit.claim_label == "header_rank"
    ]

    assert len(rank_claims) == 2
    assert rank_claims[0].claim_group is not None
    assert rank_claims[1].claim_group is not None
    assert rank_claims[0].claim_group.claim_group_id == rank_claims[1].claim_group.claim_group_id
    assert rank_claims[0].claim_group.group_kind == "field_bundle"
    assert rank_claims[0].neighbor_claim_ids == [rank_claims[1].claim_id]
    assert rank_claims[1].neighbor_claim_ids == [rank_claims[0].claim_id]


def test_loose_record_claim_units_are_low_structure_fallback_claims():
    # Loose records should reach retrieval as claim units, but with lower
    # structure quality and no invented subject. That preserves usefulness
    # without overstating confidence.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    loose_bundle = next(bundle for bundle in bundles if bundle.record_type == StructuredRecordType.LOOSE_RECORD)

    claim_units = build_claim_units_from_review_bundles([loose_bundle])

    assert len(claim_units) == 1
    assert claim_units[0].primary_subject_guess == ""
    assert claim_units[0].claim_group is not None
    assert claim_units[0].claim_group.group_kind == "prose_bundle"
    assert claim_units[0].source_family == "loose_record"
    assert claim_units[0].structure_quality < 0.5
    assert "literal_local" in claim_units[0].retrieval_channel_tags


def test_structured_review_artifact_includes_claim_units(tmp_path):
    # The JSON artifact is the bridge toward database insertion. It must carry
    # claim units alongside review bundles so later persistence work does not
    # need to rederive retrieval-shaped output from text reports.
    json_path, report_path, _llm_report_path = run_structured_review_experiment(
        "examples/world context/human history.txt",
        str(tmp_path),
        max_report_records=1,
    )

    json_text = json_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")

    assert '"claim_units"' in json_text
    assert '"document_type": "world_context"' in json_text
    assert '"document_status": "primary_canon"' in json_text
    assert '"result_level": "claim_unit"' in json_text
    assert "CLAIM UNIT SUMMARY" in report_text
    assert "document_type: world_context" in report_text
    assert "document_status: primary_canon" in report_text


def test_structured_review_artifact_preserves_document_status_manifest(tmp_path):
    # Status is provenance metadata for database insertion. It should travel
    # through diagnostics, review bundles, prompt packets, and claim units
    # without changing the record-family segmentation result.
    manifest_path = tmp_path / "metadata.json"
    manifest_path.write_text(
        """
{
  "documents": {
    "examples/world context/human history.txt": {
      "status": "historical"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    json_path, report_path, _llm_report_path = run_structured_review_experiment(
        "examples/world context/human history.txt",
        str(tmp_path),
        max_report_records=1,
        metadata_manifest_path=str(manifest_path),
    )

    json_text = json_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")

    assert '"document_status": "historical"' in json_text
    assert '"source_status": "historical"' in json_text
    assert '"source_authority_weight": 0.85' in json_text
    assert "document_status: historical" in report_text
    assert "reference_section" in report_text


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
    json_path, report_path, llm_report_path = run_structured_review_experiment(
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
    # narrow structured task plus the full deterministic seed inventory,
    # including suppressed hints that manuscript-biased filtering would
    # otherwise hide.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)

    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )

    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    assert first_bundle.llm_prompt_packet.task_name == "dossier_subject_and_explicit_facts"
    assert first_bundle.llm_prompt_packet.document_type == DocumentType.STORY_PLANNING
    assert first_bundle.llm_prompt_packet.document_status == DocumentStatus.PRIMARY_CANON
    assert first_bundle.llm_prompt_packet.source_authority == "planning_dossier"
    assert first_bundle.llm_prompt_packet.source_authority_weight == 1.0
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
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    def fake_responder(_bundle, _model):
        return ({
            "subject": {
                "subject_name": "Watanabe Yō",
                "alternate_names": ["Yō"],
                "evidence_quotes": ["WATANABE YŌ CAPTAIN / PIONEER-ADMIRAL"],
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

    updated_bundle = run_structured_llm_pass(
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


def test_completed_llm_pass_produces_separate_review_required_claim_units():
    # LLM proposals should become database-ready claim units without replacing
    # deterministic claims or being treated as canon. They need their own
    # proposal state, semantic retrieval tag, and evidence quote.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    def fake_responder(_bundle, _model):
        return ({
            "subject": {
                "subject_name": "Watanabe Yō",
                "alternate_names": ["Yō"],
                "evidence_quotes": ["WATANABE YŌ CAPTAIN / PIONEER-ADMIRAL"],
                "certainty_note": "Header line names the subject directly.",
                "unresolved": False,
            },
            "fact_proposals": [
                {
                    "label": "Role",
                    "value": "Commanding Officer, Radiant Estuary",
                    "evidence_quote": "Role: Commanding Officer, Radiant Estuary",
                    "certainty_note": "Explicit label-value line.",
                }
            ],
            "open_questions": [],
        }, "resp_test_claim_unit")

    updated_bundle = run_structured_llm_pass(
        first_bundle,
        model="gpt-4o-mini",
        responder=fake_responder,
    )

    claim_units = build_claim_units_from_review_bundles([updated_bundle])
    llm_claims = [
        claim_unit
        for claim_unit in claim_units
        if claim_unit.proposal_state.value == "llm_proposal"
    ]

    assert len(llm_claims) == 1
    assert llm_claims[0].review_state.value == "review_required"
    assert llm_claims[0].primary_subject_guess == "Watanabe Yō"
    assert llm_claims[0].claim_label == "Role"
    assert llm_claims[0].primary_evidence.quote == "Role: Commanding Officer, Radiant Estuary"
    assert llm_claims[0].retrieval_channel_tags == ["semantic_inferred"]
    assert llm_claims[0].raw_claim_payload["response_id"] == "resp_test_claim_unit"
    assert any(
        claim_unit.proposal_state.value == "deterministic_proposal"
        for claim_unit in claim_units
    )


def test_llm_report_lists_projected_llm_claim_units():
    # The separate LLM log should show the database-shaped projection, not only
    # the raw model payload. Otherwise manual inspection can miss that a
    # completed fact proposal failed to become a reviewable claim unit.
    path = "examples/story planning/estuary crew summaries.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    first_bundle = next(
        bundle for bundle in bundles
        if bundle.record_type == StructuredRecordType.DOSSIER_ENTRY
        and "WATANABE YŌ" in bundle.deterministic_seed_bundle.header_line
    )

    def fake_responder(_bundle, _model):
        return ({
            "subject": {
                "subject_name": "Watanabe Yō",
                "alternate_names": ["Yō"],
                "evidence_quotes": ["WATANABE YŌ CAPTAIN / PIONEER-ADMIRAL"],
                "certainty_note": "Header line names the subject directly.",
                "unresolved": False,
            },
            "fact_proposals": [
                {
                    "label": "Role",
                    "value": "Commanding Officer, Radiant Estuary",
                    "evidence_quote": "Role: Commanding Officer, Radiant Estuary",
                    "certainty_note": "Explicit label-value line.",
                }
            ],
            "open_questions": [],
        }, "resp_test_llm_report_claim")

    updated_bundle = run_structured_llm_pass(
        first_bundle,
        model="gpt-4o-mini",
        responder=fake_responder,
    )
    report_text = render_structured_llm_report(
        diagnostics,
        [updated_bundle],
        max_records=1,
    )

    assert "llm_claim_units:" in report_text
    assert "fact Role: Commanding Officer, Radiant Estuary" in report_text
    assert "subject=Watanabe Yō" in report_text
    assert "review=review_required" in report_text


def test_unresolved_llm_subject_does_not_force_claim_subject():
    # An LLM can extract useful facts while leaving the subject unresolved. The
    # claim unit should preserve that ambiguity instead of copying an
    # unresolved subject into the primary subject field.
    path = "examples/world context/human history.txt"
    doc, entity_records, reference_candidates = _document_outputs(path)
    records = segment_structured_records(doc)
    bundles, _diagnostics = build_structured_review_bundles(
        records,
        entity_records,
        reference_candidates,
    )
    loose_bundle = next(bundle for bundle in bundles if bundle.record_type == StructuredRecordType.LOOSE_RECORD)

    def fake_responder(_bundle, _model):
        return ({
            "subject": {
                "subject_name": "unclear",
                "alternate_names": ["Triumvirate history"],
                "evidence_quotes": [],
                "certainty_note": "The prelude is broad context rather than a subject record.",
                "unresolved": True,
            },
            "fact_proposals": [
                {
                    "label": "context_summary",
                    "value": "Human history converges from three arcs.",
                    "evidence_quote": "three long, partially overlapping arcs",
                    "certainty_note": "Explicitly stated in the prelude.",
                }
            ],
            "open_questions": ["Should this be attached to Triumvirate history?"],
        }, "resp_unresolved_subject")

    updated_bundle = run_structured_llm_pass(
        loose_bundle,
        model="gpt-4o-mini",
        responder=fake_responder,
    )
    llm_claim = next(
        claim_unit
        for claim_unit in build_claim_units_from_review_bundles([updated_bundle])
        if claim_unit.proposal_state.value == "llm_proposal"
    )

    assert llm_claim.primary_subject_guess == ""
    assert llm_claim.alternate_subject_candidates == ["Triumvirate history"]
    assert llm_claim.claim_value == "Human history converges from three arcs."


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
    # The structured scaffold writes two text logs. The main report should stay
    # focused on deterministic record structure, while the separate LLM log
    # carries model status, comparisons, and review questions.
    json_path, report_path, llm_report_path = run_structured_review_experiment(
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
