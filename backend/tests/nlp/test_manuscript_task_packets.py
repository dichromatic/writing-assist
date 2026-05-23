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
    SuppressReason,
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


def _suppressed_record(
    *,
    path: str,
    key: str,
    suppression_reason: SuppressReason,
    occurrence_count: int,
    scene_count: int,
    entityhood_score: float = 0.45,
    winning_category: LexiconCategory = LexiconCategory.UNRESOLVED,
    has_title_support: bool = False,
    attribution_count: int = 0,
) -> DocumentEntityRecord:
    """Build one suppressed entity record for rescue-selection tests."""
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(key))
    return DocumentEntityRecord(
        document_anchor=DocumentAnchor(path=path),
        normalized_key=key,
        surface_forms=[key.title()],
        winning_category=winning_category,
        resolved=False,
        entityhood_score=entityhood_score,
        entityhood_accepted=False,
        confidence_score=0.32,
        bucket=DocumentEntityBucket.SUPPRESSED,
        suppression_reason=suppression_reason,
        bucket_detail="suppressed for deterministic triage",
        occurrence_count=occurrence_count,
        rule_tier=2,
        scene_count=scene_count,
        attribution_count=attribution_count,
        has_title_support=has_title_support,
        has_possessive_support=False,
        anchors=[anchor],
        evidence_windows=[
            EvidenceWindow(
                entity_key=key,
                anchor=anchor,
                context_before="The crew called out the name in briefing.",
                context_after="Later the same term appeared again in a different scene.",
                is_first_introduction=False,
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

    packets, diagnostics = build_manuscript_task_packets(
        bundle,
        document_texts={
            "doc.md": (
                "Pioneer stood by the old estuary marker while the council reviewed logs. "
                "Later in another scene, estuary was repeated as a named place."
            )
        },
    )

    families = {packet.task_family.value for packet in packets}
    assert "manuscript_entity_profile" in families
    assert "manuscript_reference_attachment" in families
    assert "manuscript_category_resolution" in families
    assert diagnostics
    assert any(item.selected for item in diagnostics)


def test_manuscript_builder_selects_only_rescuable_suppressed_entities():
    # Rescue selection should stay narrow so only plausible suppressed entities
    # reach the LLM gate.
    record_promoted = _record("doc.md", "aldous", LexiconCategory.CHARACTER)
    rescue_ok = _suppressed_record(
        path="doc.md",
        key="pioneer",
        suppression_reason=SuppressReason.GENERIC_LEXICAL_NOISE,
        occurrence_count=4,
        scene_count=2,
        winning_category=LexiconCategory.UNRESOLVED,
        entityhood_score=0.42,
        has_title_support=True,
    )
    too_few = _suppressed_record(
        path="doc.md",
        key="aurora",
        suppression_reason=SuppressReason.LOW_ENTITYHOOD,
        occurrence_count=2,
        scene_count=2,
    )
    non_rescuable_reason = _suppressed_record(
        path="doc.md",
        key="the",
        suppression_reason=SuppressReason.STOPWORD,
        occurrence_count=8,
        scene_count=5,
    )
    too_low_entityhood = _suppressed_record(
        path="doc.md",
        key="faint",
        suppression_reason=SuppressReason.LOW_ENTITYHOOD,
        occurrence_count=5,
        scene_count=3,
        entityhood_score=0.2,
        winning_category=LexiconCategory.UNRESOLVED,
    )
    no_context = _suppressed_record(
        path="doc.md",
        key="estuary",
        suppression_reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
        occurrence_count=5,
        scene_count=2,
    )
    no_context.evidence_windows = [
        EvidenceWindow(
            entity_key="estuary",
            anchor=no_context.anchors[0],
            context_before="",
            context_after="",
            is_first_introduction=False,
            has_attribution=False,
            speaker=None,
        )
    ]

    bundle = ManuscriptReviewBundle(
        document_paths=["doc.md"],
        entity_records=[
            record_promoted,
            rescue_ok,
            too_few,
            non_rescuable_reason,
            too_low_entityhood,
            no_context,
        ],
        canonical_entities=[],
        reference_candidates=[],
        reference_clusters=[],
        conflict_records=[],
        character_summaries=[],
        review_tasks=[],
    )

    packets, diagnostics = build_manuscript_task_packets(
        bundle,
        document_texts={
            "doc.md": (
                "Pioneer stood near the old estuary marker while the council debated routes. "
                "In a later scene, estuary appeared again as a location cue."
            )
        },
    )

    rescue_packets = [
        packet for packet in packets
        if packet.task_family.value == "manuscript_suppression_rescue"
    ]
    assert {packet.source_object_id for packet in rescue_packets} == {"pioneer", "estuary"}
    pioneer_packet = next(packet for packet in rescue_packets if packet.source_object_id == "pioneer")
    assert pioneer_packet.payload["suppression_reason"] == SuppressReason.GENERIC_LEXICAL_NOISE.value
    estuary_packet = next(packet for packet in rescue_packets if packet.source_object_id == "estuary")
    # Rescue evidence now comes from assembly-time raw text slicing.
    assert len(estuary_packet.evidence_payload) == 1
    assert (
        estuary_packet.evidence_payload[0].context_before.strip()
        or estuary_packet.evidence_payload[0].context_after.strip()
    )

    rescue_diagnostics = [
        item for item in diagnostics
        if item.task_family.value == "manuscript_suppression_rescue"
    ]
    selected_ids = {item.source_object_id for item in rescue_diagnostics if item.selected}
    rejected = {item.source_object_id: item.reason for item in rescue_diagnostics if not item.selected}
    assert selected_ids == {"pioneer", "estuary"}
    assert rejected["aurora"] == "too_few_occurrences"
    assert rejected["the"].startswith("suppression_reason_stopword")
    assert rejected["faint"] == "unresolved_very_low_entityhood"
