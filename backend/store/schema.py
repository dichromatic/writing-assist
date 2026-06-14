"""
SQLite schema for the entity extraction pipeline store.

Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# -- Table definitions -------------------------------------------------------
# Kept as module-level constants so they are easy to audit in one place.

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    label TEXT,
    document_count INTEGER NOT NULL
);
"""

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    path TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL
);
"""

_CREATE_DOCUMENT_ENTITY_RECORDS = """
CREATE TABLE IF NOT EXISTS document_entity_records (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    normalized_key TEXT NOT NULL,
    document_path TEXT NOT NULL,
    bucket TEXT NOT NULL,
    winning_category TEXT NOT NULL,
    resolved INTEGER NOT NULL,
    suppression_reason TEXT,
    confidence_score REAL NOT NULL,
    entityhood_score REAL NOT NULL,
    occurrence_count INTEGER NOT NULL,
    scene_count INTEGER NOT NULL,
    classification_trace TEXT NOT NULL,
    promotion_trace TEXT NOT NULL,
    discourse_profile TEXT NOT NULL,
    support_profile TEXT NOT NULL,
    lineage_profile TEXT NOT NULL,
    source_evidence TEXT NOT NULL,
    PRIMARY KEY (run_id, normalized_key, document_path)
);
"""

_CREATE_CORPUS_ENTITIES = """
CREATE TABLE IF NOT EXISTS corpus_entities (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    canonical_key TEXT NOT NULL,
    dominant_category TEXT NOT NULL,
    aggregate_confidence REAL NOT NULL,
    review_required INTEGER NOT NULL,
    member_count INTEGER NOT NULL,
    supporting_document_count INTEGER NOT NULL,
    source_keys TEXT NOT NULL,
    conflicting_categories TEXT NOT NULL,
    reasons TEXT NOT NULL,
    absorbed_surface_forms TEXT NOT NULL,
    canonical_surface_forms TEXT NOT NULL,
    PRIMARY KEY (run_id, canonical_key)
);
"""

_CREATE_RESCUE_VERDICTS = """
CREATE TABLE IF NOT EXISTS rescue_verdicts (
    rescue_run_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    normalized_key TEXT NOT NULL,
    rescued INTEGER NOT NULL,
    entity_type TEXT,
    canonical_name TEXT,
    confidence REAL,
    rationale TEXT,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    label TEXT,
    PRIMARY KEY (rescue_run_id, normalized_key)
);
"""

# -- Indexes on hot query columns --------------------------------------------

_INDEXES = [
    """CREATE INDEX IF NOT EXISTS idx_der_run_key
       ON document_entity_records(run_id, normalized_key);""",
    """CREATE INDEX IF NOT EXISTS idx_der_run_doc
       ON document_entity_records(run_id, document_path);""",
    """CREATE INDEX IF NOT EXISTS idx_der_run_bucket
       ON document_entity_records(run_id, bucket);""",
    """CREATE INDEX IF NOT EXISTS idx_ce_run
       ON corpus_entities(run_id);""",
    """CREATE INDEX IF NOT EXISTS idx_rv_run
       ON rescue_verdicts(run_id);""",
]


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Run all CREATE TABLE and CREATE INDEX statements against *conn*.

    This is idempotent - every statement uses IF NOT EXISTS so it is safe
    to call on an already-initialised database.
    """
    conn.execute(_CREATE_RUNS)
    conn.execute(_CREATE_DOCUMENTS)
    conn.execute(_CREATE_DOCUMENT_ENTITY_RECORDS)
    conn.execute(_CREATE_CORPUS_ENTITIES)
    conn.execute(_CREATE_RESCUE_VERDICTS)
    for index_sql in _INDEXES:
        conn.execute(index_sql)


def initialize_db(db_path: Path | str) -> sqlite3.Connection:
    """Open (or create) the SQLite store and return a ready connection.

    Enables WAL mode for concurrent read access and enforces foreign key
    constraints so CASCADE deletes work correctly.

    Args:
        db_path: Filesystem path for the database file, or the literal
            string ``":memory:"`` for an in-memory database.

    Returns:
        A ``sqlite3.Connection`` with tables created and pragmas set.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _ensure_tables(conn)
    conn.commit()
    return conn
