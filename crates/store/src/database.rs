use std::fs;
use std::path::{Path, PathBuf};

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use thiserror::Error;
use writing_assist_core::ProjectConfigValidationError;

use crate::StoredMemoryKind;

const APP_STATE_DIRECTORY: &str = ".writing-assist";
const DATABASE_FILE_NAME: &str = "state.db";
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS project_directory_mappings (
    project_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    role TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    PRIMARY KEY (project_id, path),
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_candidates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_document_path TEXT NOT NULL,
    source_anchors_json TEXT NOT NULL,
    source_start_char INTEGER NOT NULL,
    source_end_char INTEGER NOT NULL,
    review_state TEXT NOT NULL,
    staleness_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviewable_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source_document_path TEXT NOT NULL,
    source_anchors_json TEXT NOT NULL,
    source_start_char INTEGER NOT NULL,
    source_end_char INTEGER NOT NULL,
    review_state TEXT NOT NULL,
    staleness_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviewable_summaries (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    text TEXT NOT NULL,
    source_document_path TEXT NOT NULL,
    source_anchors_json TEXT NOT NULL,
    source_start_char INTEGER NOT NULL,
    source_end_char INTEGER NOT NULL,
    review_state TEXT NOT NULL,
    staleness_state TEXT NOT NULL
);
"#;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    #[error(transparent)]
    Uuid(#[from] uuid::Error),
    #[error(transparent)]
    Validation(#[from] ProjectConfigValidationError),
    #[error("stored project directory role is invalid: {0}")]
    InvalidStoredDirectoryRole(String),
    #[error("stored memory review state is invalid: {0}")]
    InvalidStoredMemoryReviewState(String),
    #[error("stored memory staleness state is invalid: {0}")]
    InvalidStoredMemoryStalenessState(String),
    #[error("memory record was not found: kind={kind:?}, id={id}")]
    MemoryRecordNotFound { kind: StoredMemoryKind, id: uuid::Uuid },
}

pub fn storage_backend() -> &'static str {
    "sqlite"
}

pub fn project_database_path(project_root: &Path) -> PathBuf {
    project_root
        .join(APP_STATE_DIRECTORY)
        .join(DATABASE_FILE_NAME)
}

pub(crate) fn normalize_project_root(project_root: &Path) -> Result<PathBuf, StoreError> {
    Ok(project_root.canonicalize()?)
}

async fn open_pool(
    database_path: &Path,
    create_if_missing: bool,
) -> Result<SqlitePool, StoreError> {
    let options = SqliteConnectOptions::new()
        .filename(database_path)
        .create_if_missing(create_if_missing)
        .foreign_keys(true);

    Ok(SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?)
}

async fn initialize_schema(pool: &SqlitePool) -> Result<(), StoreError> {
    for statement in SCHEMA_SQL.split(';') {
        let sql = statement.trim();

        if sql.is_empty() {
            continue;
        }

        sqlx::query(sql).execute(pool).await?;
    }

    Ok(())
}

pub(crate) async fn open_project_database(
    project_root: &Path,
    create_if_missing: bool,
) -> Result<Option<SqlitePool>, StoreError> {
    let normalized_root = normalize_project_root(project_root)?;
    let database_path = project_database_path(&normalized_root);

    if !create_if_missing && !database_path.exists() {
        return Ok(None);
    }

    if create_if_missing {
        if let Some(parent) = database_path.parent() {
            fs::create_dir_all(parent)?;
        }
    }

    let pool = open_pool(&database_path, create_if_missing).await?;
    initialize_schema(&pool).await?;

    Ok(Some(pool))
}
