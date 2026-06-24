"""
SQLite persistence layer for pipeline extraction results.

Diagram omitted - package init with no significant information flow.
"""

from backend.store.reader import (
    get_corpus_entities_for_run,
    get_corpus_entity,
    get_document_text,
    get_records_for_document,
    get_records_for_key,
    get_records_for_run,
    get_rescue_verdicts,
    get_run,
    list_runs,
    reconstruct_evidence_context,
)
from backend.store.schema import initialize_db
from backend.store.writer import (
    create_run,
    delete_run,
    get_next_rescue_run_id,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
    persist_rescue_verdict,
)

__all__ = [
    "initialize_db",
    # Writer
    "create_run",
    "delete_run",
    "get_next_rescue_run_id",
    "persist_corpus_entities",
    "persist_document_entity_records",
    "persist_documents",
    "persist_rescue_verdict",
    # Reader
    "get_corpus_entities_for_run",
    "get_corpus_entity",
    "get_document_text",
    "get_records_for_document",
    "get_records_for_key",
    "get_records_for_run",
    "get_rescue_verdicts",
    "get_run",
    "list_runs",
    "reconstruct_evidence_context",
]
