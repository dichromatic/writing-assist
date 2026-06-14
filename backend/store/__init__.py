"""
SQLite persistence layer for pipeline extraction results.

Diagram omitted - package init with no significant information flow.
"""

from backend.store.schema import initialize_db
from backend.store.writer import (
    create_run,
    delete_run,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
)

__all__ = [
    "initialize_db",
    "create_run",
    "delete_run",
    "persist_corpus_entities",
    "persist_document_entity_records",
    "persist_documents",
]
