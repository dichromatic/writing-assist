"""Tests for the SQLite store schema, write path, and read path.

Diagram omitted - test module with no significant information flow.
"""

from __future__ import annotations

import json

from backend.nlp.types import (
    CategoryEvidenceTrace,
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
    SpanAnchor,
    SuppressedEvidence,
    SuppressReason,
)
from backend.store.reader import (
    get_records_for_document,
    get_records_for_key,
    get_records_for_run,
    reconstruct_evidence_context,
)
from backend.store.schema import initialize_db
from backend.store.writer import (
    create_run,
    delete_run,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
)


def _make_record(
    key: str = "firth",
    path: str = "doc.md",
    bucket: DocumentEntityBucket = DocumentEntityBucket.PROMOTED,
) -> DocumentEntityRecord:
    """Build a minimal but fully populated DocumentEntityRecord for tests."""
    return DocumentEntityRecord(
        identity=DocumentEntityIdentity(
            record_id="test-record-id",
            document_anchor=DocumentAnchor(path=path),
            normalized_key=key,
            surface_forms=["Firth"],
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=LexiconCategory.UNRESOLVED,
            resolved=False,
            bucket=bucket,
        ),
        source_evidence=DocumentEntitySourceEvidence(
            occurrence_count=5,
            anchors=[SpanAnchor(path=path, span_ordinal=1, start_char=10, end_char=15)],
            evidence_windows=[],
        ),
        classification_trace=DocumentEntityClassificationTrace(
            winning_score=0.45,
            runner_up_category=None,
            runner_up_score=0.0,
            evidence_by_category={
                LexiconCategory.UNRESOLVED: CategoryEvidenceTrace(
                    category=LexiconCategory.UNRESOLVED,
                    score=0.45,
                    reasons=["bare capitalized"],
                    vetoes=[],
                ),
            },
            entityhood=EntityhoodTrace(
                score=0.55,
                accepted=True,
                reasons=["multi-scene"],
                weaknesses=[],
            ),
        ),
        promotion_trace=DocumentEntityPromotionTrace(
            confidence_score=0.65,
            suppression_reason=None,
            bucket_detail="promoted",
            rule_tier=2,
            scene_count=3,
            attribution_count=0,
            possessive_count=1,
            tfidf_score=0.8,
        ),
        discourse_profile=DocumentEntityDiscourseProfile(
            in_quote_count=2,
            non_quote_count=3,
            quote_only=False,
            sentence_initial_count=1,
            sentence_initial_only=False,
            address_like_count=0,
            attributed_speaker_nearby_count=0,
            one_token_utterance_count=0,
        ),
        support_profile=DocumentEntitySupportProfile(
            title_support_count=0,
            possessive_support_count=1,
            location_support_count=0,
            linked_field_count=0,
            linked_definition_count=0,
            linked_seed_count=0,
        ),
        lineage_profile=DocumentEntityLineageProfile(
            compound_part_count=1,
            fully_covered_by_longer_compound=False,
            candidate_parent_keys=["radiant firth"],
            covered_anchor_count=0,
            uncovered_anchor_count=5,
            appears_as_compound_component=True,
            appears_as_compound_surface=False,
        ),
    )


def _make_corpus_entity(
    key: str = "firth",
    record: DocumentEntityRecord | None = None,
) -> CorpusEntity:
    """Build a minimal CorpusEntity wrapping one document record."""
    if record is None:
        record = _make_record(key=key)
    return CorpusEntity(
        canonical_key=key,
        source_keys=[key],
        member_records=[record],
        supporting_document_paths=["doc.md"],
        dominant_category=LexiconCategory.UNRESOLVED,
        aggregate_confidence=0.65,
        conflicting_categories=[],
        review_required=False,
        reasons=["single-source canonical"],
        canonical_surface_forms=["Firth"],
        absorbed_surface_forms=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_creation():
    """initialize_db on :memory: must create all five expected tables."""
    conn = initialize_db(":memory:")
    # Filter out sqlite_sequence, which SQLite creates internally when
    # AUTOINCREMENT is used on any table.
    cursor = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = sorted(row[0] for row in cursor.fetchall())
    expected = sorted([
        "runs",
        "documents",
        "document_entity_records",
        "corpus_entities",
        "rescue_verdicts",
    ])
    assert tables == expected


def test_round_trip_document_entity_record():
    """Persist a DocumentEntityRecord then read it back with raw SQL.

    Verifies that relational columns land correctly and that the JSON
    profile columns parse back into the expected structure. This encodes
    the mapping between the nested dataclass hierarchy and the flat
    relational schema - if any field path drifts, this test breaks.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)
    record = _make_record()
    persist_document_entity_records(conn, run_id, [record])

    row = conn.execute(
        "SELECT * FROM document_entity_records WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None

    # Column order matches the CREATE TABLE definition.
    (
        r_run_id, record_id, normalized_key, document_path, surface_forms,
        bucket, winning_category,
        resolved, suppression_reason, confidence_score, entityhood_score,
        occurrence_count, scene_count, classification_trace, promotion_trace,
        discourse_profile, support_profile, lineage_profile, source_evidence,
    ) = row

    assert r_run_id == run_id
    assert record_id == "test-record-id"
    assert normalized_key == "firth"
    assert document_path == "doc.md"
    assert json.loads(surface_forms) == ["Firth"]
    assert bucket == "promoted"
    assert winning_category == "unresolved"
    assert resolved == 0
    assert suppression_reason is None
    assert confidence_score == 0.65
    assert entityhood_score == 0.55
    assert occurrence_count == 5
    assert scene_count == 3

    # JSON columns must round-trip into parseable dicts/lists.
    ct = json.loads(classification_trace)
    assert ct["entityhood"]["score"] == 0.55
    assert ct["winning_score"] == 0.45

    pt = json.loads(promotion_trace)
    assert pt["confidence_score"] == 0.65
    assert pt["rule_tier"] == 2

    dp = json.loads(discourse_profile)
    assert dp["in_quote_count"] == 2

    sp = json.loads(support_profile)
    assert sp["possessive_support_count"] == 1

    lp = json.loads(lineage_profile)
    assert lp["candidate_parent_keys"] == ["radiant firth"]

    se = json.loads(source_evidence)
    assert se["occurrence_count"] == 5


def test_round_trip_corpus_entity():
    """Persist a CorpusEntity then read it back with raw SQL.

    Verifies the mapping between the CorpusEntity dataclass fields and
    the flat relational + JSON schema. Drift in member_count or
    supporting_document_count derivation surfaces here.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)
    entity = _make_corpus_entity()
    persist_corpus_entities(conn, run_id, [entity])

    row = conn.execute(
        "SELECT * FROM corpus_entities WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None

    (
        r_run_id, canonical_key, dominant_category, aggregate_confidence,
        review_required, member_count, supporting_document_count,
        source_keys, conflicting_categories, reasons,
        absorbed_surface_forms, canonical_surface_forms,
    ) = row

    assert r_run_id == run_id
    assert canonical_key == "firth"
    assert dominant_category == "unresolved"
    assert aggregate_confidence == 0.65
    assert review_required == 0
    assert member_count == 1
    assert supporting_document_count == 1
    assert json.loads(source_keys) == ["firth"]
    assert json.loads(conflicting_categories) == []
    assert json.loads(reasons) == ["single-source canonical"]
    assert json.loads(canonical_surface_forms) == ["Firth"]
    assert json.loads(absorbed_surface_forms) == []


def test_cascade_delete_removes_records():
    """Deleting a run must cascade-remove its entity records and corpus entities.

    This encodes the ON DELETE CASCADE contract. Without it, orphaned rows
    would accumulate silently and corrupt diff queries across runs.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)
    persist_document_entity_records(conn, run_id, [_make_record()])
    persist_corpus_entities(conn, run_id, [_make_corpus_entity()])

    delete_run(conn, run_id)

    der_count = conn.execute(
        "SELECT COUNT(*) FROM document_entity_records"
    ).fetchone()[0]
    ce_count = conn.execute(
        "SELECT COUNT(*) FROM corpus_entities"
    ).fetchone()[0]
    assert der_count == 0
    assert ce_count == 0


def test_document_upsert():
    """Persisting the same document path twice must keep only the latest text.

    The documents table uses INSERT OR REPLACE so re-running the pipeline
    overwrites stale text without needing explicit cleanup.
    """
    conn = initialize_db(":memory:")
    persist_documents(conn, {"doc.md": "original text"})
    persist_documents(conn, {"doc.md": "updated text"})

    row = conn.execute("SELECT raw_text FROM documents WHERE path = 'doc.md'").fetchone()
    assert row is not None
    assert row[0] == "updated text"

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1


def test_cascade_delete_removes_rescue_verdicts():
    """Deleting a run must cascade-remove associated rescue verdict rows.

    Rescue verdicts reference a run_id with ON DELETE CASCADE. This test
    manually inserts a verdict row (bypassing the writer, which does not
    yet cover rescue persistence) and confirms the cascade fires.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)

    # Insert a rescue verdict directly - the writer does not cover this yet.
    conn.execute(
        "INSERT INTO rescue_verdicts "
        "(rescue_run_id, run_id, normalized_key, rescued, model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (1, run_id, "firth", 1, "qwen3.6", "2025-01-01T00:00:00Z"),
    )
    conn.commit()

    delete_run(conn, run_id)

    count = conn.execute("SELECT COUNT(*) FROM rescue_verdicts").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# Read-path tests
# ---------------------------------------------------------------------------


def test_get_records_for_key_across_documents():
    """Records with the same key in different documents must all be returned.

    This encodes the cross-document query path that corpus reconciliation
    depends on. If the index or query silently drops documents, entity
    merging becomes lossy.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 2)

    rec_a = _make_record(key="firth", path="chapter1.md")
    rec_b = _make_record(key="firth", path="chapter2.md")
    persist_document_entity_records(conn, run_id, [rec_a, rec_b])

    results = get_records_for_key(conn, run_id, "firth")

    assert len(results) == 2
    paths = sorted(r.identity.document_anchor.path for r in results)
    assert paths == ["chapter1.md", "chapter2.md"]

    # Verify nested profiles survived deserialisation.
    for r in results:
        assert isinstance(r, DocumentEntityRecord)
        assert r.classification_trace.entityhood.score == 0.55
        assert r.promotion_trace.confidence_score == 0.65
        assert r.lineage_profile.candidate_parent_keys == ["radiant firth"]


def test_get_records_for_document_across_keys():
    """Records with different keys in the same document must all be returned.

    This encodes the per-document listing that the inspection view uses.
    Missing records would silently hide entities from the document report.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)

    rec_a = _make_record(key="firth", path="doc.md")
    rec_b = _make_record(key="radiant firth", path="doc.md")
    persist_document_entity_records(conn, run_id, [rec_a, rec_b])

    results = get_records_for_document(conn, run_id, "doc.md")

    assert len(results) == 2
    keys = sorted(r.identity.normalized_key for r in results)
    assert keys == ["firth", "radiant firth"]

    for r in results:
        assert isinstance(r, DocumentEntityRecord)
        assert r.discourse_profile.in_quote_count == 2


def test_evidence_context_reconstruction():
    """Context reconstruction must slice stored text correctly around a mention.

    This is the primary way the UI shows entity evidence without shipping
    the full document. Off-by-one errors here would corrupt the display.
    """
    conn = initialize_db(":memory:")
    text = "The Radiant Firth stretched across the valley like a silver ribbon."
    persist_documents(conn, {"story.md": text})

    # "Radiant Firth" starts at index 4, ends at index 17.
    result = reconstruct_evidence_context(
        conn, "story.md", start_char=4, end_char=17, context_window=10
    )

    assert result["mention"] == "Radiant Firth"
    assert result["context_before"] == "The "
    assert result["context_after"] == " stretched"


def test_empty_result_sets():
    """Querying non-existent run/key/document must return empty lists, not errors.

    Callers rely on empty-list semantics to distinguish 'no results' from
    errors. Raising an exception here would break iteration patterns.
    """
    conn = initialize_db(":memory:")

    assert get_records_for_run(conn, 999) == []
    assert get_records_for_key(conn, 999, "nonexistent") == []
    assert get_records_for_document(conn, 999, "nonexistent.md") == []

    # Evidence context for a missing document returns empty strings.
    ctx = reconstruct_evidence_context(conn, "missing.md", 0, 10)
    assert ctx == {"context_before": "", "mention": "", "context_after": ""}


def test_record_deserialisation_fidelity():
    """A record with all profile fields populated must round-trip exactly.

    This is the critical deserialisation drift test. Every nested profile
    field is checked so that if the writer serialises differently from what
    the reader expects, the test breaks immediately rather than silently
    corrupting downstream consumers.
    """
    conn = initialize_db(":memory:")
    run_id = create_run(conn, "2025-01-01T00:00:00Z", "abc1234", 1)

    # Build a maximally populated record to stress every deserialisation path.
    record = DocumentEntityRecord(
        identity=DocumentEntityIdentity(
            record_id="test-full-record",
            document_anchor=DocumentAnchor(path="novel.md"),
            normalized_key="firth",
            surface_forms=["Firth", "the Firth"],
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=LexiconCategory.UNRESOLVED,
            resolved=False,
            bucket=DocumentEntityBucket.SUPPRESSED,
        ),
        source_evidence=DocumentEntitySourceEvidence(
            occurrence_count=7,
            anchors=[
                SpanAnchor(path="novel.md", span_ordinal=3, start_char=100, end_char=105),
                SpanAnchor(path="novel.md", span_ordinal=8, start_char=400, end_char=405),
            ],
            evidence_windows=[
                EvidenceWindow(
                    entity_key="firth",
                    anchor=SpanAnchor(
                        path="novel.md", span_ordinal=3, start_char=100, end_char=105
                    ),
                    context_before="the old ",
                    context_after=", standing tall",
                    is_first_introduction=True,
                    has_attribution=False,
                    speaker=None,
                ),
            ],
            suppressed_related_evidence=[
                SuppressedEvidence(
                    document_anchor=DocumentAnchor(path="novel.md"),
                    normalized_key="old firth",
                    surface_forms=["Old Firth"],
                    winning_category=LexiconCategory.CHARACTER,
                    confidence_score=0.3,
                    reason=SuppressReason.COMPONENT_OVERLAP_NOISE,
                    detail="fully covered by longer compound",
                    anchors=[
                        SpanAnchor(
                            path="novel.md", span_ordinal=3, start_char=96, end_char=105
                        ),
                    ],
                ),
            ],
        ),
        classification_trace=DocumentEntityClassificationTrace(
            winning_score=0.45,
            runner_up_category=LexiconCategory.CHARACTER,
            runner_up_score=0.30,
            evidence_by_category={
                LexiconCategory.UNRESOLVED: CategoryEvidenceTrace(
                    category=LexiconCategory.UNRESOLVED,
                    score=0.45,
                    reasons=["bare capitalized", "multi-scene"],
                    vetoes=["quote-heavy"],
                ),
                LexiconCategory.CHARACTER: CategoryEvidenceTrace(
                    category=LexiconCategory.CHARACTER,
                    score=0.30,
                    reasons=["possessive form"],
                    vetoes=[],
                ),
            },
            entityhood=EntityhoodTrace(
                score=0.40,
                accepted=False,
                reasons=["multi-scene"],
                weaknesses=["quote-only discourse", "no structural support"],
            ),
        ),
        promotion_trace=DocumentEntityPromotionTrace(
            confidence_score=0.35,
            suppression_reason=SuppressReason.LOW_ENTITYHOOD,
            bucket_detail="entityhood too weak",
            rule_tier=1,
            scene_count=2,
            attribution_count=1,
            possessive_count=3,
            tfidf_score=0.6,
        ),
        discourse_profile=DocumentEntityDiscourseProfile(
            in_quote_count=5,
            non_quote_count=2,
            quote_only=False,
            sentence_initial_count=3,
            sentence_initial_only=False,
            address_like_count=1,
            attributed_speaker_nearby_count=2,
            one_token_utterance_count=1,
        ),
        support_profile=DocumentEntitySupportProfile(
            title_support_count=1,
            possessive_support_count=3,
            location_support_count=2,
            linked_field_count=1,
            linked_definition_count=0,
            linked_seed_count=1,
        ),
        lineage_profile=DocumentEntityLineageProfile(
            compound_part_count=1,
            fully_covered_by_longer_compound=False,
            candidate_parent_keys=["radiant firth"],
            covered_anchor_count=1,
            uncovered_anchor_count=6,
            appears_as_compound_component=True,
            appears_as_compound_surface=False,
        ),
    )

    persist_document_entity_records(conn, run_id, [record])
    results = get_records_for_key(conn, run_id, "firth")
    assert len(results) == 1
    r = results[0]

    # -- identity --
    assert r.identity.record_id == "test-full-record"
    assert r.identity.normalized_key == "firth"
    assert r.identity.document_anchor.path == "novel.md"
    assert r.identity.surface_forms == ["Firth", "the Firth"]

    # -- current_state --
    assert r.current_state.bucket == DocumentEntityBucket.SUPPRESSED
    assert r.current_state.winning_category == LexiconCategory.UNRESOLVED
    assert r.current_state.resolved is False

    # -- classification_trace --
    ct = r.classification_trace
    assert ct.winning_score == 0.45
    assert ct.runner_up_category == LexiconCategory.CHARACTER
    assert ct.runner_up_score == 0.30
    assert LexiconCategory.UNRESOLVED in ct.evidence_by_category
    assert LexiconCategory.CHARACTER in ct.evidence_by_category

    unresolved_ev = ct.evidence_by_category[LexiconCategory.UNRESOLVED]
    assert unresolved_ev.score == 0.45
    assert "bare capitalized" in unresolved_ev.reasons
    assert "quote-heavy" in unresolved_ev.vetoes

    char_ev = ct.evidence_by_category[LexiconCategory.CHARACTER]
    assert char_ev.score == 0.30
    assert "possessive form" in char_ev.reasons

    assert ct.entityhood.score == 0.40
    assert ct.entityhood.accepted is False
    assert "quote-only discourse" in ct.entityhood.weaknesses

    # -- promotion_trace --
    pt = r.promotion_trace
    assert pt.confidence_score == 0.35
    assert pt.suppression_reason == SuppressReason.LOW_ENTITYHOOD
    assert pt.bucket_detail == "entityhood too weak"
    assert pt.rule_tier == 1
    assert pt.scene_count == 2
    assert pt.attribution_count == 1
    assert pt.possessive_count == 3
    assert pt.tfidf_score == 0.6

    # -- discourse_profile --
    dp = r.discourse_profile
    assert dp.in_quote_count == 5
    assert dp.non_quote_count == 2
    assert dp.quote_only is False
    assert dp.sentence_initial_count == 3
    assert dp.sentence_initial_only is False
    assert dp.address_like_count == 1
    assert dp.attributed_speaker_nearby_count == 2
    assert dp.one_token_utterance_count == 1

    # -- support_profile --
    sp = r.support_profile
    assert sp.title_support_count == 1
    assert sp.possessive_support_count == 3
    assert sp.location_support_count == 2
    assert sp.linked_field_count == 1
    assert sp.linked_definition_count == 0
    assert sp.linked_seed_count == 1

    # -- lineage_profile --
    lp = r.lineage_profile
    assert lp.compound_part_count == 1
    assert lp.fully_covered_by_longer_compound is False
    assert lp.candidate_parent_keys == ["radiant firth"]
    assert lp.covered_anchor_count == 1
    assert lp.uncovered_anchor_count == 6
    assert lp.appears_as_compound_component is True
    assert lp.appears_as_compound_surface is False

    # -- source_evidence --
    se = r.source_evidence
    assert se.occurrence_count == 7
    assert len(se.anchors) == 2
    assert se.anchors[0].start_char == 100
    assert se.anchors[1].start_char == 400

    # Evidence windows must survive the round-trip.
    assert len(se.evidence_windows) == 1
    ew = se.evidence_windows[0]
    assert ew.entity_key == "firth"
    assert ew.context_before == "the old "
    assert ew.context_after == ", standing tall"
    assert ew.is_first_introduction is True
    assert ew.has_attribution is False
    assert ew.speaker is None

    # Suppressed related evidence must survive the round-trip.
    assert len(se.suppressed_related_evidence) == 1
    sup = se.suppressed_related_evidence[0]
    assert sup.normalized_key == "old firth"
    assert sup.winning_category == LexiconCategory.CHARACTER
    assert sup.reason == SuppressReason.COMPONENT_OVERLAP_NOISE
    assert sup.confidence_score == 0.3
    assert len(sup.anchors) == 1
