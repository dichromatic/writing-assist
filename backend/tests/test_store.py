"""Tests for the SQLite store schema and write path.

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
    LexiconCategory,
    SpanAnchor,
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
        r_run_id, normalized_key, document_path, bucket, winning_category,
        resolved, suppression_reason, confidence_score, entityhood_score,
        occurrence_count, scene_count, classification_trace, promotion_trace,
        discourse_profile, support_profile, lineage_profile, source_evidence,
    ) = row

    assert r_run_id == run_id
    assert normalized_key == "firth"
    assert document_path == "doc.md"
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
