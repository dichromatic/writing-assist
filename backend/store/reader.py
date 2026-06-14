"""
Read-path helpers for querying pipeline output from the SQLite store.

Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

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


# ---------------------------------------------------------------------------
# Run queries
# ---------------------------------------------------------------------------


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    """Return all pipeline runs as plain dicts.

    Args:
        conn: Active database connection.

    Returns:
        List of dicts with keys: run_id, created_at, git_commit, label,
        document_count. Ordered by run_id ascending.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT run_id, created_at, git_commit, label, document_count "
        "FROM runs ORDER BY run_id"
    ).fetchall()
    conn.row_factory = None
    return [dict(row) for row in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    """Return a single run as a dict, or None if it does not exist.

    Args:
        conn: Active database connection.
        run_id: The run to look up.

    Returns:
        Dict with run metadata, or None.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT run_id, created_at, git_commit, label, document_count "
        "FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.row_factory = None
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Document queries
# ---------------------------------------------------------------------------


def get_document_text(conn: sqlite3.Connection, path: str) -> str | None:
    """Return raw text for a document path, or None if not stored.

    Args:
        conn: Active database connection.
        path: Document path as stored in the documents table.

    Returns:
        The raw document text, or None.
    """
    row = conn.execute(
        "SELECT raw_text FROM documents WHERE path = ?", (path,)
    ).fetchone()
    if row is None:
        return None
    return row[0]


# ---------------------------------------------------------------------------
# Document entity record queries
# ---------------------------------------------------------------------------


def get_records_for_run(
    conn: sqlite3.Connection, run_id: int
) -> list[DocumentEntityRecord]:
    """Return all document entity records for a run as typed Python objects.

    Args:
        conn: Active database connection.
        run_id: The run to query.

    Returns:
        List of fully deserialised DocumentEntityRecord objects.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM document_entity_records WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.row_factory = None
    return [_row_to_record(row) for row in rows]


def get_records_for_key(
    conn: sqlite3.Connection, run_id: int, normalized_key: str
) -> list[DocumentEntityRecord]:
    """Return all records for a normalized key across documents in one run.

    Args:
        conn: Active database connection.
        run_id: The run to query.
        normalized_key: The entity key to filter on.

    Returns:
        List of fully deserialised DocumentEntityRecord objects.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM document_entity_records "
        "WHERE run_id = ? AND normalized_key = ?",
        (run_id, normalized_key),
    ).fetchall()
    conn.row_factory = None
    return [_row_to_record(row) for row in rows]


def get_records_for_document(
    conn: sqlite3.Connection, run_id: int, document_path: str
) -> list[DocumentEntityRecord]:
    """Return all records for a document path in one run.

    Args:
        conn: Active database connection.
        run_id: The run to query.
        document_path: The document path to filter on.

    Returns:
        List of fully deserialised DocumentEntityRecord objects.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM document_entity_records "
        "WHERE run_id = ? AND document_path = ?",
        (run_id, document_path),
    ).fetchall()
    conn.row_factory = None
    return [_row_to_record(row) for row in rows]


# ---------------------------------------------------------------------------
# Corpus entity queries
# ---------------------------------------------------------------------------


def get_corpus_entities_for_run(
    conn: sqlite3.Connection, run_id: int
) -> list[CorpusEntity]:
    """Return all corpus entities for a run as typed Python objects.

    member_records is set to an empty list because actual member records
    live in the document_entity_records table. The caller reconstructs
    membership if needed by joining on normalized_key.

    Args:
        conn: Active database connection.
        run_id: The run to query.

    Returns:
        List of CorpusEntity objects with empty member_records.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM corpus_entities WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.row_factory = None
    return [_row_to_corpus_entity(row) for row in rows]


def get_corpus_entity(
    conn: sqlite3.Connection, run_id: int, canonical_key: str
) -> CorpusEntity | None:
    """Return a single corpus entity, or None if it does not exist.

    Args:
        conn: Active database connection.
        run_id: The run to query.
        canonical_key: The canonical key to look up.

    Returns:
        A CorpusEntity with empty member_records, or None.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM corpus_entities WHERE run_id = ? AND canonical_key = ?",
        (run_id, canonical_key),
    ).fetchone()
    conn.row_factory = None
    if row is None:
        return None
    return _row_to_corpus_entity(row)


# ---------------------------------------------------------------------------
# Evidence context
# ---------------------------------------------------------------------------


def reconstruct_evidence_context(
    conn: sqlite3.Connection,
    path: str,
    start_char: int,
    end_char: int,
    context_window: int = 100,
) -> dict:
    """Slice stored document text around a mention span for display.

    Reconstructs a context window without requiring the caller to hold the
    full document text in memory. Returns empty strings for all fields if
    the document is not found.

    Args:
        conn: Active database connection.
        path: Document path in the documents table.
        start_char: Inclusive start offset of the mention.
        end_char: Exclusive end offset of the mention.
        context_window: Number of characters of context on each side.

    Returns:
        Dict with keys context_before, mention, context_after.
    """
    text = get_document_text(conn, path)
    if text is None:
        return {"context_before": "", "mention": "", "context_after": ""}

    before_start = max(0, start_char - context_window)
    after_end = min(len(text), end_char + context_window)

    return {
        "context_before": text[before_start:start_char],
        "mention": text[start_char:end_char],
        "context_after": text[end_char:after_end],
    }


# ---------------------------------------------------------------------------
# Rescue verdict queries
# ---------------------------------------------------------------------------


def get_rescue_verdicts(
    conn: sqlite3.Connection,
    run_id: int,
    rescue_run_id: int | None = None,
) -> list[dict]:
    """Return rescue verdicts for a run as plain dicts.

    Args:
        conn: Active database connection.
        run_id: The pipeline run whose verdicts to retrieve.
        rescue_run_id: If provided, filter to a specific rescue run.

    Returns:
        List of dicts with verdict fields. The rescued field is converted
        from integer to bool for ergonomic use in Python.
    """
    conn.row_factory = sqlite3.Row
    if rescue_run_id is not None:
        rows = conn.execute(
            "SELECT * FROM rescue_verdicts "
            "WHERE run_id = ? AND rescue_run_id = ?",
            (run_id, rescue_run_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM rescue_verdicts WHERE run_id = ?", (run_id,)
        ).fetchall()
    conn.row_factory = None

    results = []
    for row in rows:
        d = dict(row)
        # Convert the integer flag to a proper bool.
        d["rescued"] = bool(d["rescued"])
        return_keys = [
            "rescue_run_id", "run_id", "normalized_key", "rescued",
            "entity_type", "canonical_name", "confidence", "rationale",
            "model", "created_at", "label",
        ]
        results.append({k: d[k] for k in return_keys})
    return results


# ---------------------------------------------------------------------------
# Private deserialisation helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: sqlite3.Row) -> DocumentEntityRecord:
    """Convert a database row into a fully typed DocumentEntityRecord.

    The JSON profile columns are parsed and reconstructed into the nested
    dataclass hierarchy so callers receive the same types the writer
    serialised from.

    Args:
        row: A sqlite3.Row from the document_entity_records table.

    Returns:
        A complete DocumentEntityRecord with all nested profiles.
    """
    classification_data = json.loads(row["classification_trace"])
    promotion_data = json.loads(row["promotion_trace"])
    discourse_data = json.loads(row["discourse_profile"])
    support_data = json.loads(row["support_profile"])
    lineage_data = json.loads(row["lineage_profile"])
    source_data = json.loads(row["source_evidence"])

    return DocumentEntityRecord(
        identity=DocumentEntityIdentity(
            record_id=row["record_id"],
            document_anchor=DocumentAnchor(path=row["document_path"]),
            normalized_key=row["normalized_key"],
            surface_forms=json.loads(row["surface_forms"]),
        ),
        current_state=DocumentEntityCurrentState(
            winning_category=LexiconCategory(row["winning_category"]),
            resolved=bool(row["resolved"]),
            bucket=DocumentEntityBucket(row["bucket"]),
        ),
        source_evidence=_deserialise_source_evidence(source_data),
        classification_trace=_deserialise_classification_trace(classification_data),
        promotion_trace=_deserialise_promotion_trace(promotion_data),
        discourse_profile=_deserialise_discourse_profile(discourse_data),
        support_profile=_deserialise_support_profile(support_data),
        lineage_profile=_deserialise_lineage_profile(lineage_data),
    )


def _deserialise_classification_trace(
    data: dict,
) -> DocumentEntityClassificationTrace:
    """Reconstruct a classification trace from its JSON representation.

    Args:
        data: Parsed JSON dict from the classification_trace column.

    Returns:
        A typed DocumentEntityClassificationTrace.
    """
    evidence_by_category = {
        LexiconCategory(k): CategoryEvidenceTrace(
            category=LexiconCategory(v["category"]),
            score=v["score"],
            reasons=v["reasons"],
            vetoes=v["vetoes"],
        )
        for k, v in data["evidence_by_category"].items()
    }

    runner_up = None
    if data["runner_up_category"] is not None:
        runner_up = LexiconCategory(data["runner_up_category"])

    return DocumentEntityClassificationTrace(
        winning_score=data["winning_score"],
        runner_up_category=runner_up,
        runner_up_score=data["runner_up_score"],
        evidence_by_category=evidence_by_category,
        entityhood=EntityhoodTrace(
            score=data["entityhood"]["score"],
            accepted=data["entityhood"]["accepted"],
            reasons=data["entityhood"]["reasons"],
            weaknesses=data["entityhood"]["weaknesses"],
        ),
    )


def _deserialise_promotion_trace(data: dict) -> DocumentEntityPromotionTrace:
    """Reconstruct a promotion trace from its JSON representation.

    Args:
        data: Parsed JSON dict from the promotion_trace column.

    Returns:
        A typed DocumentEntityPromotionTrace.
    """
    suppression_reason: Optional[SuppressReason] = None
    if data["suppression_reason"] is not None:
        suppression_reason = SuppressReason(data["suppression_reason"])

    return DocumentEntityPromotionTrace(
        confidence_score=data["confidence_score"],
        suppression_reason=suppression_reason,
        bucket_detail=data["bucket_detail"],
        rule_tier=data["rule_tier"],
        scene_count=data["scene_count"],
        attribution_count=data["attribution_count"],
        possessive_count=data["possessive_count"],
        tfidf_score=data["tfidf_score"],
    )


def _deserialise_discourse_profile(
    data: dict,
) -> DocumentEntityDiscourseProfile:
    """Reconstruct a discourse profile from its JSON representation.

    Args:
        data: Parsed JSON dict from the discourse_profile column.

    Returns:
        A typed DocumentEntityDiscourseProfile.
    """
    return DocumentEntityDiscourseProfile(
        in_quote_count=data["in_quote_count"],
        non_quote_count=data["non_quote_count"],
        quote_only=data["quote_only"],
        sentence_initial_count=data["sentence_initial_count"],
        sentence_initial_only=data["sentence_initial_only"],
        address_like_count=data["address_like_count"],
        attributed_speaker_nearby_count=data["attributed_speaker_nearby_count"],
        one_token_utterance_count=data["one_token_utterance_count"],
    )


def _deserialise_support_profile(data: dict) -> DocumentEntitySupportProfile:
    """Reconstruct a support profile from its JSON representation.

    Args:
        data: Parsed JSON dict from the support_profile column.

    Returns:
        A typed DocumentEntitySupportProfile.
    """
    return DocumentEntitySupportProfile(
        title_support_count=data["title_support_count"],
        possessive_support_count=data["possessive_support_count"],
        location_support_count=data["location_support_count"],
        linked_field_count=data["linked_field_count"],
        linked_definition_count=data["linked_definition_count"],
        linked_seed_count=data["linked_seed_count"],
    )


def _deserialise_lineage_profile(data: dict) -> DocumentEntityLineageProfile:
    """Reconstruct a lineage profile from its JSON representation.

    Args:
        data: Parsed JSON dict from the lineage_profile column.

    Returns:
        A typed DocumentEntityLineageProfile.
    """
    return DocumentEntityLineageProfile(
        compound_part_count=data["compound_part_count"],
        fully_covered_by_longer_compound=data["fully_covered_by_longer_compound"],
        candidate_parent_keys=data["candidate_parent_keys"],
        covered_anchor_count=data["covered_anchor_count"],
        uncovered_anchor_count=data["uncovered_anchor_count"],
        appears_as_compound_component=data["appears_as_compound_component"],
        appears_as_compound_surface=data["appears_as_compound_surface"],
    )


def _deserialise_source_evidence(data: dict) -> DocumentEntitySourceEvidence:
    """Reconstruct source evidence from its JSON representation.

    Handles the nested SpanAnchor, EvidenceWindow, and SuppressedEvidence
    lists that are serialised as JSON arrays of dicts.

    Args:
        data: Parsed JSON dict from the source_evidence column.

    Returns:
        A typed DocumentEntitySourceEvidence.
    """
    anchors = [
        SpanAnchor(
            path=a["path"],
            span_ordinal=a["span_ordinal"],
            start_char=a["start_char"],
            end_char=a["end_char"],
        )
        for a in data["anchors"]
    ]

    evidence_windows = [
        EvidenceWindow(
            entity_key=w["entity_key"],
            anchor=SpanAnchor(
                path=w["anchor"]["path"],
                span_ordinal=w["anchor"]["span_ordinal"],
                start_char=w["anchor"]["start_char"],
                end_char=w["anchor"]["end_char"],
            ),
            context_before=w["context_before"],
            context_after=w["context_after"],
            is_first_introduction=w["is_first_introduction"],
            has_attribution=w["has_attribution"],
            speaker=w["speaker"],
        )
        for w in data["evidence_windows"]
    ]

    suppressed = [
        SuppressedEvidence(
            document_anchor=DocumentAnchor(path=s["document_anchor"]["path"]),
            normalized_key=s["normalized_key"],
            surface_forms=s["surface_forms"],
            winning_category=LexiconCategory(s["winning_category"]),
            confidence_score=s["confidence_score"],
            reason=SuppressReason(s["reason"]),
            detail=s["detail"],
            anchors=[
                SpanAnchor(
                    path=a["path"],
                    span_ordinal=a["span_ordinal"],
                    start_char=a["start_char"],
                    end_char=a["end_char"],
                )
                for a in s["anchors"]
            ],
        )
        for s in data.get("suppressed_related_evidence", [])
    ]

    return DocumentEntitySourceEvidence(
        occurrence_count=data["occurrence_count"],
        anchors=anchors,
        evidence_windows=evidence_windows,
        suppressed_related_evidence=suppressed,
    )


def _row_to_corpus_entity(row: sqlite3.Row) -> CorpusEntity:
    """Convert a database row into a typed CorpusEntity.

    member_records and supporting_document_paths are set to empty lists
    because the actual data lives in the document_entity_records table.
    The caller can reconstruct membership by joining on source_keys.

    Args:
        row: A sqlite3.Row from the corpus_entities table.

    Returns:
        A CorpusEntity with empty member_records and supporting_document_paths.
    """
    return CorpusEntity(
        canonical_key=row["canonical_key"],
        source_keys=json.loads(row["source_keys"]),
        member_records=[],
        supporting_document_paths=[],
        dominant_category=LexiconCategory(row["dominant_category"]),
        aggregate_confidence=row["aggregate_confidence"],
        conflicting_categories=[
            LexiconCategory(c) for c in json.loads(row["conflicting_categories"])
        ],
        review_required=bool(row["review_required"]),
        reasons=json.loads(row["reasons"]),
        canonical_surface_forms=json.loads(row["canonical_surface_forms"]),
        absorbed_surface_forms=json.loads(row["absorbed_surface_forms"]),
    )
