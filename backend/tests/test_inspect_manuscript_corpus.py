"""Tests for the manuscript corpus inspection artifact contract."""

import json
from pathlib import Path

from inspect_manuscript_corpus import _write_manuscript_artifacts
from backend.nlp.semantic_review import render_manuscript_review_report
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictSource,
    CorpusEntity,
    DocumentEntityClassificationTrace,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityCurrentState,
    DocumentEntityDiscourseProfile,
    DocumentEntityRecord,
    DocumentEntityIdentity,
    DocumentEntityLineageProfile,
    DocumentEntityPromotionTrace,
    DocumentEntitySourceEvidence,
    DocumentEntitySupportProfile,
    EvidenceWindow,
    EntityhoodTrace,
    CategoryEvidenceTrace,
    LexiconCategory,
    ManuscriptReviewBundle,
    ReviewTask,
    ReviewTaskKind,
    SpanAnchor,
    SuppressReason,
)


def _make_record(
    path: str,
    normalized_key: str,
    category: LexiconCategory,
    *,
    bucket: DocumentEntityBucket,
    entityhood_weaknesses: list[str] | None = None,
    suppression_reason: SuppressReason | None = None,
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
        identity=DocumentEntityIdentity(
            record_id=f"record:{path}:{normalized_key}",
            document_anchor=DocumentAnchor(path=path),
            normalized_key=normalized_key,
            surface_forms=[normalized_key.title()],
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=category,
            resolved=True,
            bucket=bucket,
        ),
        source_evidence=DocumentEntitySourceEvidence(
            occurrence_count=1,
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
        ),
        classification_trace=DocumentEntityClassificationTrace(
            winning_score=0.7,
            runner_up_category=None,
            runner_up_score=0.0,
            evidence_by_category={
                category: CategoryEvidenceTrace(
                    category=category,
                    score=0.7,
                    reasons=[],
                    vetoes=[],
                )
            },
            entityhood=EntityhoodTrace(
                score=0.7,
                accepted=True,
                reasons=[],
                weaknesses=list(entityhood_weaknesses or []),
            ),
        ),
        promotion_trace=DocumentEntityPromotionTrace(
            confidence_score=0.7,
            suppression_reason=suppression_reason,
            bucket_detail="",
            rule_tier=1,
            scene_count=1,
            attribution_count=0,
            possessive_count=0,
            tfidf_score=0.0,
        ),
        discourse_profile=DocumentEntityDiscourseProfile(
            in_quote_count=0,
            non_quote_count=1,
            quote_only=False,
            sentence_initial_count=0,
            sentence_initial_only=False,
            address_like_count=0,
            attributed_speaker_nearby_count=0,
            one_token_utterance_count=0,
        ),
        support_profile=DocumentEntitySupportProfile(
            title_support_count=0,
            possessive_support_count=0,
            location_support_count=0,
            linked_field_count=0,
            linked_definition_count=0,
            linked_seed_count=0,
        ),
        lineage_profile=DocumentEntityLineageProfile(
            compound_part_count=1,
            fully_covered_by_longer_compound=False,
            candidate_parent_keys=[],
            covered_anchor_count=0,
            uncovered_anchor_count=1,
            appears_as_compound_component=False,
            appears_as_compound_surface=False,
        ),
    )


def test_report_treats_buckets_as_visibility_tiers_and_stops_at_questions():
    # The manuscript handoff should explain bucket semantics explicitly and
    # stop at review questions in the primary report. If proposal-shaped
    # output returns here, later semantic review will inherit misleadingly
    # assertive framing even when the underlying evidence is ambiguous.
    records = [
        _make_record("doc.md", "aldous", LexiconCategory.CHARACTER, bucket=DocumentEntityBucket.PROMOTED),
        _make_record(
            "doc.md",
            "captain",
            LexiconCategory.CHARACTER,
            bucket=DocumentEntityBucket.SUPPRESSED,
            suppression_reason=SuppressReason.QUOTE_ONLY_ADDRESS_LIKE_DISCOURSE,
        ),
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
    assert "DOCUMENT ENTITY RECORD SNAPSHOT" in report
    assert "Selected examples expose the nested record contract in readable form." in report
    assert "suppressed  : hidden from the main entity inventory, but retained" in report
    assert "identity: record_id=record:doc.md:aldous" in report
    assert "class: win=0.700  runner_up=-  entityhood=0.700  accepted=yes" in report
    assert "promo: conf=0.700  tier=1  scenes=1  attr=0  poss=0  tfidf=0.000  suppression=quote_only_address_like_discourse" in report
    assert "support: title=0  poss=0  loc=0  linked_fields=0  linked_definitions=0  linked_seeds=0" in report
    assert "discourse: quote=0  non_quote=1  quote_only=no" in report
    assert "lineage: parts=1  fully_covered=no  covered=0  uncovered=1" in report
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

    report_path, json_path, llm_task_report_path = _write_manuscript_artifacts(
        bundle,
        document_texts={"doc.md": "Captain Aldous addressed the bridge crew."},
        output_path=str(tmp_path / "manuscript-report.txt"),
    )

    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    llm_task_report = llm_task_report_path.read_text(encoding="utf-8")
    assert json_path == tmp_path / "manuscript-report.json"
    assert artifact["artifact_version"] == "2"
    assert artifact["source_kind"] == "manuscript"
    assert artifact["review_bundle"]["canonical_entities"][0]["absorbed_surface_forms"] == ["Captain Aldous"]
    assert artifact["review_bundle"]["review_tasks"][0]["ranked_speaker_keys"] == ["kohaku"]
    assert "llm_task_packets" in artifact
    assert "llm_task_diagnostics" in artifact
    assert "LLM TASK PACKETS" in llm_task_report
    assert "Semantic handoff stops at review questions in this report." in report
    assert "🚢" not in report
