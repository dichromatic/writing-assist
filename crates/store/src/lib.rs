mod database;
mod memory;
mod projects;

pub use database::{project_database_path, storage_backend, StoreError};
pub use memory::{
    list_entity_candidates, list_reviewable_facts, list_reviewable_summaries,
    mark_memory_stale_for_document, save_entity_candidates, save_reviewable_facts,
    save_reviewable_summaries, update_memory_review_state, MemoryRecordFilter, StoredMemoryKind,
};
pub use projects::{load_project_config, save_project_config};

pub(crate) use database::{normalize_project_root, open_project_database};
