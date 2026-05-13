"""Tests for manuscript task-packet generation from manuscript review bundles."""

from backend.nlp.llm_tasks.assembly.manuscripts import build_manuscript_task_packets
from backend.nlp.types import (
    CharacterSemanticSummary,
    ConflictRecord,
    ConflictSource,
    CorpusEntity,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityRecord,
    EvidenceWindow,
    LexiconCategory,
    ManuscriptReviewBundle,
    ReferenceCandidateType,
    ReferenceCluster,
    ReviewTask,
    ReviewTaskKind,
    SpanAnchor,
)


def _record(path: str, key: str, category: LexiconCategory) -> DocumentEntityRecord:
    """Build a minimal document entity record for manuscript task tests."""
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(key))
    return DocumentEntityRecord(
        document_anchor=DocumentAnchor(path=path),
        normalized_key=key,
        surface_forms=[key.title()],
        winning_category=category,
        resolved=True,
        entityhood_score=0.8,
        entityhood_accepted=True,
        confidence_score=0.8,
        bucket=DocumentEntityBucket.PROMOTED,
        suppression_reason=None,
        bucket_detail="",
        occurrence_count=2,
        rule_tier=1,
        scene_count=1,
        attribution_count=0,
        has_title_support=False,
        has_possessive_support=False,
        anchors=[anchor],
        evidence_windows=[
            EvidenceWindow(
                entity_key=key,
                anchor=anchor,
                context_before="",
                context_after="",
                is_first_introduction=True,
                has_attribution=False,
                speaker=None,
            )
        ],
    )


def test_manuscript_builder_emits_entity_reference_and_conflict_task_families():
    # Manuscript handoff should generate shared task families from deterministic
    # bundle content before any provider execution.
    record = _record("doc.md", "aldous", LexiconCategory.CHARACTER)
    entity = CorpusEntity(
        canonical_key="aldous",
        source_keys=["aldous", "captain aldous"],
        member_records=[record],
        supporting_document_paths=["doc.md", "doc2.md"],
        dominant_category=LexiconCategory.CHARACTER,
        aggregate_confidence=0.8,
        conflicting_categories=[],
        review_required=False,
        reasons=[],
        absorbed_surface_forms=["Captain Aldous"],
    )
    reference_cluster = ReferenceCluster(
        document_anchor=DocumentAnchor(path="doc.md"),
        reference_type=ReferenceCandidateType.BARE_TITLE_ROLE,
        normalized="captain",
        surface_forms=["Captain"],
        occurrence_count=2,
        anchors=[record.anchors[0]],
        in_quote_count=0,
        address_like_count=0,
        speaker_entity_scores={"aldous": 1},
        candidate_entity_scores={"aldous": 2},
    )
    conflict = ConflictRecord(
        canonical_key="aldous",
        source=ConflictSource.SURFACE_LEVEL_DISAGREEMENT,
        conflicting_categories=[LexiconCategory.CHARACTER, LexiconCategory.GROUP],
        supporting_document_paths=["doc.md"],
        reason="Cross-document category disagreement.",
    )
    bundle = ManuscriptReviewBundle(
        document_paths=["doc.md", "doc2.md"],
        entity_records=[record],
        canonical_entities=[entity],
        reference_candidates=[],
        reference_clusters=[reference_cluster],
        conflict_records=[conflict],
        character_summaries=[
            CharacterSemanticSummary(
                canonical_key="aldous",
                alias_keys=[],
                supporting_document_paths=["doc.md"],
                attached_title_counts={"captain": 1},
                ambiguous_title_counts={},
                attached_relation_counts={},
                ambiguous_relation_counts={},
                aggregate_attribution_count=0,
                conflict_sources=[],
            )
        ],
        review_tasks=[
            ReviewTask(
                task_id="task-1",
                kind=ReviewTaskKind.TITLE_ROLE_ATTACHMENT,
                subject_key="captain",
                prompt="Does captain attach to aldous?",
                supporting_anchor_paths=["doc.md"],
            )
        ],
    )

    packets, diagnostics = build_manuscript_task_packets(bundle)

    families = {packet.task_family.value for packet in packets}
    assert "manuscript_entity_profile" in families
    assert "manuscript_reference_attachment" in families
    assert "manuscript_category_resolution" in families
    assert diagnostics
    assert any(item.selected for item in diagnostics)
