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
            record.identity.normalized_key,
            record.identity.document_anchor.path,
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
        "  run_id, normalized_key, document_path, bucket, winning_category,"
        "  resolved, suppression_reason, confidence_score, entityhood_score,"
        "  occurrence_count, scene_count, classification_trace, promotion_trace,"
        "  discourse_profile, support_profile, lineage_profile, source_evidence"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
