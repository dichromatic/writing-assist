"""
Write-path helpers for persisting pipeline output to the SQLite store.

Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations

import json
import sqlite3

from backend.nlp.text_filtering import to_llm_safe_jsonable
from backend.nlp.types import CorpusEntity, DocumentEntityRecord


def create_run(
    conn: sqlite3.Connection,
    created_at: str,
    git_commit: str,
    document_count: int,
    label: str | None = None,
) -> int:
    """Insert a new pipeline run row and return the generated run_id.

    Args:
        conn: Active database connection.
        created_at: ISO 8601 timestamp for the run.
        git_commit: Short git hash identifying the code version.
        document_count: Number of documents processed in this run.
        label: Optional human-readable note.

    Returns:
        The auto-incremented ``run_id`` for the new row.
    """
    cursor = conn.execute(
        "INSERT INTO runs (created_at, git_commit, label, document_count) "
        "VALUES (?, ?, ?, ?)",
        (created_at, git_commit, label, document_count),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def persist_documents(conn: sqlite3.Connection, documents: dict[str, str]) -> None:
    """Upsert document source text by path.

    Uses INSERT OR REPLACE so repeated runs overwrite previous text for the
    same document path without requiring an explicit DELETE first.

    Args:
        conn: Active database connection.
        documents: Mapping of document path to raw text content.
    """
    conn.executemany(
        "INSERT OR REPLACE INTO documents (path, raw_text) VALUES (?, ?)",
        list(documents.items()),
    )
    conn.commit()


def persist_document_entity_records(
    conn: sqlite3.Connection,
    run_id: int,
    records: list[DocumentEntityRecord],
) -> None:
    """Serialise and insert document entity records for a given run.

    Nested profile dataclasses are converted to JSON via
    ``to_llm_safe_jsonable`` so the store captures the full reasoning
    trace without coupling the schema to Python dataclass structure.

    Args:
        conn: Active database connection.
        run_id: The run these records belong to.
        records: Document-local entity records to persist.
    """
    rows = []
    for record in records:
        # Derive the suppression reason string, or None when not suppressed.
        suppression_reason = (
            record.promotion_trace.suppression_reason.value
            if record.promotion_trace.suppression_reason is not None
            else None
        )

        rows.append((
            run_id,
            record.identity.record_id,
            record.identity.normalized_key,
            record.identity.document_anchor.path,
            json.dumps(record.identity.surface_forms),
            record.current_state.bucket.value,
            record.current_state.winning_category.value,
            int(record.current_state.resolved),
            suppression_reason,
            record.promotion_trace.confidence_score,
            record.classification_trace.entityhood.score,
            record.source_evidence.occurrence_count,
            record.promotion_trace.scene_count,
            json.dumps(to_llm_safe_jsonable(record.classification_trace)),
            json.dumps(to_llm_safe_jsonable(record.promotion_trace)),
            json.dumps(to_llm_safe_jsonable(record.discourse_profile)),
            json.dumps(to_llm_safe_jsonable(record.support_profile)),
            json.dumps(to_llm_safe_jsonable(record.lineage_profile)),
            json.dumps(to_llm_safe_jsonable(record.source_evidence)),
        ))

    conn.executemany(
        "INSERT INTO document_entity_records ("
        "  run_id, record_id, normalized_key, document_path, surface_forms,"
        "  bucket, winning_category,"
        "  resolved, suppression_reason, confidence_score, entityhood_score,"
        "  occurrence_count, scene_count, classification_trace, promotion_trace,"
        "  discourse_profile, support_profile, lineage_profile, source_evidence"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def persist_corpus_entities(
    conn: sqlite3.Connection,
    run_id: int,
    entities: list[CorpusEntity],
) -> None:
    """Serialise and insert corpus-level canonical entities for a given run.

    JSON list columns are serialised with ``json.dumps`` so the store
    preserves the full merge evidence without requiring separate join
    tables.

    Args:
        conn: Active database connection.
        run_id: The run these entities belong to.
        entities: Corpus-level canonical entities to persist.
    """
    rows = []
    for entity in entities:
        rows.append((
            run_id,
            entity.canonical_key,
            entity.dominant_category.value,
            entity.aggregate_confidence,
            int(entity.review_required),
            len(entity.member_records),
            len(entity.supporting_document_paths),
            json.dumps(entity.source_keys),
            json.dumps([c.value for c in entity.conflicting_categories]),
            json.dumps(entity.reasons),
            json.dumps(entity.absorbed_surface_forms),
            json.dumps(entity.canonical_surface_forms),
        ))

    conn.executemany(
        "INSERT INTO corpus_entities ("
        "  run_id, canonical_key, dominant_category, aggregate_confidence,"
        "  review_required, member_count, supporting_document_count,"
        "  source_keys, conflicting_categories, reasons,"
        "  absorbed_surface_forms, canonical_surface_forms"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def persist_rescue_verdict(
    conn: sqlite3.Connection,
    *,
    rescue_run_id: int,
    run_id: int,
    normalized_key: str,
    rescued: bool,
    model: str,
    created_at: str,
    label: str | None = None,
    entity_type: str | None = None,
    canonical_name: str | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
) -> None:
    """Insert one rescue verdict row.

    Each verdict records the LLM's binary decision for a single
    normalized key within a rescue run. Multiple verdicts with
    different normalized_keys share the same rescue_run_id.

    Args:
        conn: Active database connection.
        rescue_run_id: Sequence number for this rescue run.
        run_id: The extraction run these verdicts target.
        normalized_key: The entity key being evaluated.
        rescued: Whether the LLM judged this entity as genuine.
        model: Model identifier used for the verdict.
        created_at: ISO 8601 timestamp for the verdict.
        label: Optional human-readable label for the rescue run.
        entity_type: Optional LLM-provided type hint.
        canonical_name: Optional LLM-provided display name.
        confidence: Optional LLM self-reported confidence.
        rationale: Optional LLM rationale for the verdict.
    """
    conn.execute(
        "INSERT INTO rescue_verdicts ("
        "  rescue_run_id, run_id, normalized_key, rescued, entity_type,"
        "  canonical_name, confidence, rationale, model, created_at, label"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rescue_run_id,
            run_id,
            normalized_key,
            int(rescued),
            entity_type,
            canonical_name,
            confidence,
            rationale,
            model,
            created_at,
            label,
        ),
    )
    conn.commit()


def get_next_rescue_run_id(conn: sqlite3.Connection, run_id: int) -> int:
    """Return the next available rescue_run_id for a given extraction run.

    Computes max(rescue_run_id) + 1 from existing verdicts for this
    run_id, or 1 if no rescue runs exist yet.

    Args:
        conn: Active database connection.
        run_id: The extraction run to query.

    Returns:
        The next rescue run sequence number.
    """
    row = conn.execute(
        "SELECT MAX(rescue_run_id) FROM rescue_verdicts WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    current_max = row[0] if row[0] is not None else 0
    return current_max + 1


def delete_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Delete a pipeline run and all dependent rows.

    Foreign key CASCADE constraints handle removing document entity
    records, corpus entities, and rescue verdicts that reference this run.

    Args:
        conn: Active database connection.
        run_id: The run to remove.
    """
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    conn.commit()
