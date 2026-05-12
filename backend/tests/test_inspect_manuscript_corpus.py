"""Tests for the manuscript corpus inspection artifact contract."""

import json
from pathlib import Path

from inspect_manuscript_corpus import _write_manuscript_artifacts
from backend.nlp.semantic_review import render_manuscript_review_report
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictSource,
    CorpusEntity,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    EvidenceWindow,
    LexiconCategory,
    ManuscriptReviewBundle,
    ReviewTask,
    ReviewTaskKind,
    SpanAnchor,
)


def _make_record(
    path: str,
    normalized_key: str,
    category: LexiconCategory,
    *,
    bucket: DocumentEntityBucket,
) -> DocumentEntityRecord:
    """Build a minimal document entity record for report tests.

    Args:
        path: Source document path.
        normalized_key: Normalized document-local entity key.
        category: Winning deterministic category.
        bucket: Presentation bucket to expose in the report.

    Returns:
        A minimal stable document entity record.
    """
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(normalized_key))
    return DocumentEntityRecord(
        document_anchor=DocumentAnchor(path=path),
        normalized_key=normalized_key,
        surface_forms=[normalized_key.title()],
        winning_category=category,
        resolved=True,
        entityhood_score=0.7,
        entityhood_accepted=True,
        confidence_score=0.7,
        bucket=bucket,
        suppression_reason=None,
        bucket_detail="",
        occurrence_count=1,
        rule_tier=1,
        scene_count=1,
        attribution_count=0,
        has_title_support=False,
        has_possessive_support=False,
        anchors=[anchor],
        evidence_windows=[
            EvidenceWindow(
                entity_key=normalized_key,
                anchor=anchor,
                context_before="",
                context_after="",
                is_first_introduction=True,
                has_attribution=False,
                speaker=None,
            )
        ],
    )


def test_report_treats_buckets_as_visibility_tiers_and_stops_at_questions():
    # The manuscript handoff should explain bucket semantics explicitly and
    # stop at review questions in the primary report. If proposal-shaped
    # output returns here, later semantic review will inherit misleadingly
    # assertive framing even when the underlying evidence is ambiguous.
    records = [
        _make_record("doc.md", "aldous", LexiconCategory.CHARACTER, bucket=DocumentEntityBucket.PROMOTED),
        _make_record("doc.md", "captain", LexiconCategory.CHARACTER, bucket=DocumentEntityBucket.SUPPRESSED),
    ]
    review_tasks = [
        ReviewTask(
            task_id="task-1",
            kind=ReviewTaskKind.TITLE_ROLE_ATTACHMENT,
            subject_key="captain",
            prompt="Does the bare title 'captain' refer to aldous?",
            supporting_anchor_paths=["doc.md"],
            ranked_candidate_keys=["aldous", "beatrix"],
            ranked_speaker_keys=["kohaku"],
            corpus_owner_keys=["aldous"],
            evidence_note="local candidates preserved; quote speakers preserved",
        )
    ]
    character_summaries = [
        CharacterSemanticSummary(
            canonical_key="aldous",
            alias_keys=[],
            supporting_document_paths=["doc.md"],
            attached_title_counts={"captain": 1},
            ambiguous_title_counts={},
            attached_relation_counts={},
            ambiguous_relation_counts={},
            aggregate_attribution_count=0,
            conflict_sources=[ConflictSource.COMPONENT_POLLUTION],
            canonical_surface_forms=["Aldous"],
            absorbed_surface_forms=["🚢 Captain Aldous"],
            merge_reasons=["titled or role-led compound deferred to stronger personal key"],
        )
    ]
    bundle = ManuscriptReviewBundle(
        document_paths=["doc.md"],
        entity_records=records,
        canonical_entities=[
            CorpusEntity(
                canonical_key="aldous",
                source_keys=["aldous"],
                member_records=[records[0]],
                supporting_document_paths=["doc.md"],
                dominant_category=LexiconCategory.CHARACTER,
                aggregate_confidence=0.7,
                conflicting_categories=[],
                review_required=False,
                reasons=[],
            )
        ],
        reference_candidates=[],
        reference_clusters=[],
        conflict_records=[],
        character_summaries=character_summaries,
        review_tasks=review_tasks,
    )

    report = render_manuscript_review_report(bundle)

    assert "BUCKET SEMANTICS" in report
    assert "suppressed  : hidden from the main entity inventory, but retained" in report
    assert "Semantic handoff stops at review questions in this report." in report
    assert "SEMANTIC PROPOSALS" not in report
    assert "ranked_candidates: aldous, beatrix" in report
    assert "ranked_speakers: kohaku" in report
    assert "corpus_owners: aldous" in report
    assert "absorbed_surfaces: Captain Aldous" in report
    assert "🚢" not in report
    assert "merge_reasons: titled or role-led compound deferred to stronger personal key" in report


def test_artifact_writer_persists_machine_readable_bundle_next_to_report(tmp_path: Path):
    # The manuscript handoff needs a persisted artifact, not just a text
    # report, so later semantic stages can consume the same structured
    # evidence the report displays. If this silently stops writing JSON, the
    # report can still look correct while the machine-readable handoff breaks.
    record = _make_record(
        "doc.md",
        "aldous",
        LexiconCategory.CHARACTER,
        bucket=DocumentEntityBucket.PROMOTED,
    )
    bundle = ManuscriptReviewBundle(
        document_paths=["doc.md"],
        entity_records=[record],
        canonical_entities=[
            CorpusEntity(
                canonical_key="aldous",
                source_keys=["aldous"],
                member_records=[record],
                supporting_document_paths=["doc.md"],
                dominant_category=LexiconCategory.CHARACTER,
                aggregate_confidence=0.7,
                conflicting_categories=[],
                review_required=False,
                reasons=[],
                canonical_surface_forms=["Aldous"],
                absorbed_surface_forms=["🚢 Captain Aldous"],
            )
        ],
        reference_candidates=[],
        reference_clusters=[],
        conflict_records=[],
        character_summaries=[],
        review_tasks=[
            ReviewTask(
                task_id="task-1",
                kind=ReviewTaskKind.TITLE_ROLE_ATTACHMENT,
                subject_key="captain",
                prompt="Does the bare title 'captain' refer to aldous?",
                supporting_anchor_paths=["doc.md"],
                ranked_candidate_keys=["aldous"],
                ranked_speaker_keys=["kohaku"],
                corpus_owner_keys=["aldous"],
                evidence_note="local candidates preserved; quote speakers preserved",
            )
        ],
    )

    report_path, json_path = _write_manuscript_artifacts(
        bundle,
        str(tmp_path / "manuscript-report.txt"),
    )

    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert json_path == tmp_path / "manuscript-report.json"
    assert artifact["canonical_entities"][0]["absorbed_surface_forms"] == ["Captain Aldous"]
    assert artifact["review_tasks"][0]["ranked_speaker_keys"] == ["kohaku"]
    assert "Semantic handoff stops at review questions in this report." in report
    assert "🚢" not in report
