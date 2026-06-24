"""Tests for the CLI entrypoint run-corpus pipeline-to-store integration.

Diagram omitted - test module with no significant information flow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from backend.nlp.pipeline import run_corpus_pipeline
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.store import (
    create_run,
    initialize_db,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
)

# A minimal manuscript fragment with enough structure for the pipeline to
# produce entity records. Firth appears as a dialogue speaker with a
# possessive-like pattern, and the Radiant appears as a named vessel in
# narrative context. Both should survive as at least document-level records.
_FIXTURE_TEXT = """\
# Chapter One

Captain Firth stood on the bridge of the Radiant. The ship hummed quietly.

"We should head north," Firth said.

Firth turned to the navigator. The Radiant was their only hope.
"""


def test_run_corpus_populates_store():
    """The run-corpus flow must persist documents, entity records, and corpus entities.

    This is an integration test covering the full path from raw Markdown files
    through the NLP pipeline, corpus reconciliation, and into the SQLite store.
    It verifies that the store contains at least one row in each of the three
    output tables. Without this test, a wiring mistake between pipeline output
    and store persistence would be invisible until a real corpus run.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_dir = Path(tmp_dir) / "corpus"
        corpus_dir.mkdir()

        doc_path = corpus_dir / "chapter_one.md"
        doc_path.write_text(_FIXTURE_TEXT, encoding="utf-8")

        db_path = Path(tmp_dir) / "test.db"
        path_strings = [str(doc_path)]

        # -- Pipeline --
        corpus_result = run_corpus_pipeline(path_strings)
        reconciliation_result = reconcile_document_entities(
            corpus_result.entity_records
        )

        # -- Persistence --
        conn = initialize_db(db_path)
        try:
            run_id = create_run(
                conn, "2025-01-01T00:00:00Z", "test1234", len(path_strings)
            )

            documents_dict = {
                str(doc_path): doc_path.read_text(encoding="utf-8")
            }
            persist_documents(conn, documents_dict)
            persist_document_entity_records(
                conn, run_id, corpus_result.entity_records
            )
            persist_corpus_entities(
                conn, run_id, reconciliation_result.canonical_entities
            )

            # -- Verify documents table --
            doc_count = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            assert doc_count == 1, (
                f"Expected 1 document row, got {doc_count}"
            )

            # -- Verify document entity records table --
            der_count = conn.execute(
                "SELECT COUNT(*) FROM document_entity_records WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            assert der_count >= 1, (
                f"Expected at least 1 document entity record, got {der_count}"
            )

            # -- Verify corpus entities table --
            ce_count = conn.execute(
                "SELECT COUNT(*) FROM corpus_entities WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            assert ce_count >= 1, (
                f"Expected at least 1 corpus entity, got {ce_count}"
            )

            # -- Verify the run row itself --
            run_row = conn.execute(
                "SELECT document_count, git_commit FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert run_row is not None
            assert run_row[0] == 1
            assert run_row[1] == "test1234"

        finally:
            conn.close()
