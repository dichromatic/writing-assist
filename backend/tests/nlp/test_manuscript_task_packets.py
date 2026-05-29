"""Tests for suppression rescue task-packet generation from manuscript review bundles.

These tests encode the deterministic rescue gate: which suppressed entities
qualify for LLM verification and which are correctly rejected. The gate
keeps rescue traffic narrow (~30-50 candidates out of ~800+ suppressed)
so the LLM pass stays cheap and targeted.
"""

from backend.nlp.llm_tasks.rescue import build_rescue_task_packets
from backend.nlp.types import (
    CategoryEvidenceTrace,
    CharacterSemanticSummary,
    CorpusEntity,
    DocumentAnchor,
    DocumentEntityBucket,
    DocumentEntityClassificationTrace,
    DocumentEntityCurrentState,
    DocumentEntityDiscourseProfile,
    DocumentEntityIdentity,
    DocumentEntityLineageProfile,
    DocumentEntityPromotionTrace,
    DocumentEntityRecord,
    DocumentEntitySourceEvidence,
    DocumentEntitySupportProfile,
    EntityhoodTrace,
    EvidenceWindow,
    LexiconCategory,
    ManuscriptReviewBundle,
    SpanAnchor,
    SuppressReason,
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
    in_quote_count: int = 0,
    non_quote_count: int | None = None,
    quote_only: bool = False,
    address_like_count: int = 0,
    one_token_utterance_count: int = 0,
    fully_covered_by_longer_compound: bool = False,
    candidate_parent_keys: list[str] | None = None,
    covered_anchor_count: int = 0,
    uncovered_anchor_count: int = 1,
    appears_as_compound_component: bool = False,
) -> DocumentEntityRecord:
    """Build one suppressed entity record for rescue-selection tests."""
    anchor = SpanAnchor(path=path, span_ordinal=0, start_char=0, end_char=len(key))
    return DocumentEntityRecord(
        identity=DocumentEntityIdentity(
            record_id=f"test-{path}-{key}",
            document_anchor=DocumentAnchor(path=path),
            normalized_key=key,
            surface_forms=[key.title()],
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=winning_category,
            resolved=False,
            bucket=DocumentEntityBucket.SUPPRESSED,
        ),
        source_evidence=DocumentEntitySourceEvidence(
            occurrence_count=occurrence_count,
            anchors=[anchor],
            evidence_windows=[],
        ),
        classification_trace=DocumentEntityClassificationTrace(
            winning_score=0.32,
            runner_up_category=None,
            runner_up_score=0.0,
            evidence_by_category={
                winning_category: CategoryEvidenceTrace(
                    category=winning_category,
                    score=0.32,
                    reasons=[],
                    vetoes=[],
                )
            },
            entityhood=EntityhoodTrace(
                score=entityhood_score,
                accepted=False,
                reasons=[],
                weaknesses=[],
            ),
        ),
        promotion_trace=DocumentEntityPromotionTrace(
            confidence_score=0.32,
            suppression_reason=suppression_reason,
            bucket_detail="suppressed for deterministic triage",
            rule_tier=2,
            scene_count=scene_count,
            attribution_count=attribution_count,
            possessive_count=0,
            tfidf_score=0.0,
        ),
        discourse_profile=DocumentEntityDiscourseProfile(
            in_quote_count=in_quote_count,
            non_quote_count=occurrence_count if non_quote_count is None else non_quote_count,
            quote_only=quote_only,
            sentence_initial_count=0,
            sentence_initial_only=False,
            address_like_count=address_like_count,
            attributed_speaker_nearby_count=0,
            one_token_utterance_count=one_token_utterance_count,
        ),
        support_profile=DocumentEntitySupportProfile(
            title_support_count=1 if has_title_support else 0,
            possessive_support_count=0,
            location_support_count=0,
            linked_field_count=0,
            linked_definition_count=0,
            linked_seed_count=0,
        ),
        lineage_profile=DocumentEntityLineageProfile(
            compound_part_count=len(key.split()),
            fully_covered_by_longer_compound=fully_covered_by_longer_compound,
            candidate_parent_keys=[] if candidate_parent_keys is None else candidate_parent_keys,
            covered_anchor_count=covered_anchor_count,
            uncovered_anchor_count=uncovered_anchor_count,
            appears_as_compound_component=appears_as_compound_component,
            appears_as_compound_surface=(" " in key),
        ),
    )


def _bundle_with_records(
    records: list[DocumentEntityRecord],
) -> ManuscriptReviewBundle:
    """Build a minimal manuscript bundle containing only entity records."""
    return ManuscriptReviewBundle(
        document_paths=["doc.md"],
        entity_records=records,
        canonical_entities=[],
        reference_candidates=[],
        reference_clusters=[],
        conflict_records=[],
        character_summaries=[],
        review_tasks=[],
    )


_DOC_TEXTS = {
    "doc.md": (
        "Pioneer stood near the old estuary marker while the council debated routes. "
        "In a later scene, estuary appeared again as a location cue. "
        "The crew discussed pioneer's legacy and its significance."
    )
}


def test_rescue_selects_only_rescuable_suppressed_entities():
    # Rescue selection should stay narrow so only plausible suppressed
    # entities reach the LLM gate. Stopwords, singletons, and very low
    # entityhood records must be rejected.
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
        occurrence_count=1,
        scene_count=1,
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

    bundle = _bundle_with_records([rescue_ok, too_few, non_rescuable_reason, too_low_entityhood])
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    selected_ids = {p.source_object_id for p in packets}
    assert selected_ids == {"pioneer"}

    rejected = {d.source_object_id: d.reason for d in diagnostics if not d.selected}
    assert rejected["aurora"] == "too_few_occurrences"
    assert rejected["the"].startswith("suppression_reason_stopword")
    assert rejected["faint"] == "unresolved_very_low_entityhood"


def test_rescue_builds_evidence_from_raw_text():
    # Rescue evidence is built at assembly time from raw document text,
    # not from promotion-stage evidence windows (which are empty for
    # suppressed records).
    record = _suppressed_record(
        path="doc.md",
        key="estuary",
        suppression_reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
        occurrence_count=5,
        scene_count=2,
    )
    bundle = _bundle_with_records([record])
    packets, _ = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert len(packets) == 1
    packet = packets[0]
    assert len(packet.evidence_payload) >= 1
    evidence = packet.evidence_payload[0]
    assert evidence.context_before.strip() or evidence.context_after.strip()


def test_rescue_skips_when_document_text_missing():
    # When document text is not available for a candidate, it should
    # be rejected rather than producing an empty-evidence packet.
    record = _suppressed_record(
        path="missing.md",
        key="phantom",
        suppression_reason=SuppressReason.LOW_ENTITYHOOD,
        occurrence_count=5,
        scene_count=3,
    )
    bundle = _bundle_with_records([record])
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert len(packets) == 0
    rejected = {d.source_object_id: d.reason for d in diagnostics if not d.selected}
    assert rejected["phantom"] == "missing_document_text"


def test_rescue_packet_payload_contains_entity_metadata():
    # The task packet payload should carry enough deterministic metadata
    # for the LLM to make an informed verdict.
    record = _suppressed_record(
        path="doc.md",
        key="pioneer",
        suppression_reason=SuppressReason.GENERIC_LEXICAL_NOISE,
        occurrence_count=6,
        scene_count=3,
        entityhood_score=0.5,
    )
    bundle = _bundle_with_records([record])
    packets, _ = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert len(packets) == 1
    payload = packets[0].payload
    assert payload["normalized_key"] == "pioneer"
    assert payload["suppression_reasons"] == ["generic_lexical_noise"]
    assert payload["occurrence_count"] == 6
    assert payload["scene_count"] == 3
    assert payload["entityhood_score"] == 0.5


def test_rescue_rejects_quote_only_address_like_discourse_junk():
    # Quote-only unresolved address-like tokens are the clearest low-value
    # rescue calls and should be rejected before they reach the LLM.
    record = _suppressed_record(
        path="doc.md",
        key="hey",
        suppression_reason=SuppressReason.GENERIC_LEXICAL_NOISE,
        occurrence_count=5,
        scene_count=3,
        in_quote_count=5,
        non_quote_count=0,
        quote_only=True,
        address_like_count=3,
        one_token_utterance_count=1,
        winning_category=LexiconCategory.UNRESOLVED,
        entityhood_score=0.38,
    )
    bundle = _bundle_with_records([record])
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert packets == []
    rejected = {d.source_object_id: d.reason for d in diagnostics if not d.selected}
    assert rejected["hey"] == "quote_only_address_like_discourse"


def test_rescue_keeps_component_fragment_with_uncovered_support():
    # Component-overlap fragments still deserve LLM triage when they retain
    # uncovered anchors outside longer compounds.
    record = _suppressed_record(
        path="doc.md",
        key="firth",
        suppression_reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
        occurrence_count=5,
        scene_count=3,
        covered_anchor_count=3,
        uncovered_anchor_count=2,
        appears_as_compound_component=True,
        candidate_parent_keys=["radiant firth"],
    )
    bundle = _bundle_with_records([record])
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert {packet.source_object_id for packet in packets} == {"firth"}
    selected = {d.source_object_id: d.reason for d in diagnostics if d.selected}
    assert selected["firth"] == "rescue_candidate"


def test_rescue_rejects_fully_covered_component_fragment_without_uncovered_support():
    # Fully covered component fragments add no new local evidence and should
    # not survive rescue triage just because they recur inside longer forms.
    record = _suppressed_record(
        path="doc.md",
        key="radiant",
        suppression_reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
        occurrence_count=6,
        scene_count=3,
        fully_covered_by_longer_compound=True,
        candidate_parent_keys=["radiant firth", "radiant estuary"],
        covered_anchor_count=6,
        uncovered_anchor_count=0,
        appears_as_compound_component=True,
    )
    bundle = _bundle_with_records([record])
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    assert packets == []
    rejected = {d.source_object_id: d.reason for d in diagnostics if not d.selected}
    assert rejected["radiant"] == "fully_covered_by_longer_compound"


def test_rescue_skips_entities_already_absorbed_into_compounds():
    # If reconciliation already absorbed "estuary" into compound "radiant estuary",
    # there is no point asking the LLM whether "estuary" is a standalone entity.
    estuary_record = _suppressed_record(
        path="doc.md",
        key="estuary",
        suppression_reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
        occurrence_count=8,
        scene_count=4,
    )
    pioneer_record = _suppressed_record(
        path="doc.md",
        key="pioneer",
        suppression_reason=SuppressReason.GENERIC_LEXICAL_NOISE,
        occurrence_count=5,
        scene_count=3,
    )
    compound = CorpusEntity(
        canonical_key="radiant estuary",
        source_keys=["radiant estuary", "estuary"],
        member_records=[],
        supporting_document_paths=["doc.md"],
        dominant_category=LexiconCategory.PLACE,
        aggregate_confidence=0.8,
        conflicting_categories=[],
        review_required=False,
        reasons=["contained alias absorbed"],
    )
    bundle = ManuscriptReviewBundle(
        document_paths=["doc.md"],
        entity_records=[estuary_record, pioneer_record],
        canonical_entities=[compound],
        reference_candidates=[],
        reference_clusters=[],
        conflict_records=[],
        character_summaries=[],
        review_tasks=[],
    )
    packets, diagnostics = build_rescue_task_packets(bundle, _DOC_TEXTS)

    selected_ids = {p.source_object_id for p in packets}
    assert "estuary" not in selected_ids
    assert "pioneer" in selected_ids
    rejected = {d.source_object_id: d.reason for d in diagnostics if not d.selected}
    assert rejected["estuary"] == "already_absorbed_into_compound"
