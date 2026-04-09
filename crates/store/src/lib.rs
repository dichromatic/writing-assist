use std::fs;
use std::path::{Path, PathBuf};

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use thiserror::Error;
use writing_assist_core::{
    validate_project_directory_mappings, ProjectConfig, ProjectConfigValidationError,
    ProjectDirectoryMapping, ProjectDirectoryRole,
};

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
"#;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Validation(#[from] ProjectConfigValidationError),
    #[error("stored project directory role is invalid: {0}")]
    InvalidStoredDirectoryRole(String),
}

pub fn storage_backend() -> &'static str {
    "sqlite"
}

pub fn project_database_path(project_root: &Path) -> PathBuf {
    project_root
        .join(APP_STATE_DIRECTORY)
        .join(DATABASE_FILE_NAME)
}

fn role_to_storage_value(role: &ProjectDirectoryRole) -> &'static str {
    match role {
        ProjectDirectoryRole::PrimaryManuscript => "primary_manuscript",
        ProjectDirectoryRole::Reference => "reference",
        ProjectDirectoryRole::Notes => "notes",
        ProjectDirectoryRole::Ignore => "ignore",
    }
}

fn role_from_storage_value(value: &str) -> Result<ProjectDirectoryRole, StoreError> {
    match value {
        "primary_manuscript" => Ok(ProjectDirectoryRole::PrimaryManuscript),
        "reference" => Ok(ProjectDirectoryRole::Reference),
        "notes" => Ok(ProjectDirectoryRole::Notes),
        "ignore" => Ok(ProjectDirectoryRole::Ignore),
        _ => Err(StoreError::InvalidStoredDirectoryRole(value.to_string())),
    }
}

fn normalize_project_root(project_root: &Path) -> Result<PathBuf, StoreError> {
    Ok(project_root.canonicalize()?)
}

async fn open_pool(database_path: &Path, create_if_missing: bool) -> Result<SqlitePool, StoreError> {
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

pub async fn save_project_config(
    project_root: &Path,
    directory_mappings: &[ProjectDirectoryMapping],
) -> Result<ProjectConfig, StoreError> {
    validate_project_directory_mappings(directory_mappings)?;

    let normalized_root = normalize_project_root(project_root)?;
    let database_path = project_database_path(&normalized_root);

    if let Some(parent) = database_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let pool = open_pool(&database_path, true).await?;
    initialize_schema(&pool).await?;

    let normalized_root_string = normalized_root.to_string_lossy().to_string();
    let mut transaction = pool.begin().await?;

    sqlx::query(
        "INSERT INTO projects (root_path) VALUES (?) ON CONFLICT(root_path) DO NOTHING",
    )
    .bind(&normalized_root_string)
    .execute(&mut *transaction)
    .await?;

    let project_row = sqlx::query("SELECT id FROM projects WHERE root_path = ?")
        .bind(&normalized_root_string)
        .fetch_one(&mut *transaction)
        .await?;
    let project_id: i64 = project_row.get("id");

    sqlx::query("DELETE FROM project_directory_mappings WHERE project_id = ?")
        .bind(project_id)
        .execute(&mut *transaction)
        .await?;

    for mapping in directory_mappings {
        // Phase 1.4 persists the reviewed import configuration as the source of truth for later discovery.
        sqlx::query(
            "INSERT INTO project_directory_mappings (project_id, path, role, enabled) VALUES (?, ?, ?, ?)",
        )
        .bind(project_id)
        .bind(&mapping.path)
        .bind(role_to_storage_value(&mapping.role))
        .bind(mapping.enabled)
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;

    Ok(ProjectConfig {
        root_path: normalized_root_string,
        directory_mappings: directory_mappings.to_vec(),
    })
}

pub async fn load_project_config(project_root: &Path) -> Result<Option<ProjectConfig>, StoreError> {
    let normalized_root = normalize_project_root(project_root)?;
    let database_path = project_database_path(&normalized_root);

    if !database_path.exists() {
        return Ok(None);
    }

    let pool = open_pool(&database_path, false).await?;
    initialize_schema(&pool).await?;

    let normalized_root_string = normalized_root.to_string_lossy().to_string();
    let Some(project_row) = sqlx::query("SELECT id, root_path FROM projects WHERE root_path = ?")
        .bind(&normalized_root_string)
        .fetch_optional(&pool)
        .await?
    else {
        return Ok(None);
    };

    let project_id: i64 = project_row.get("id");
    let root_path: String = project_row.get("root_path");

    let mapping_rows = sqlx::query(
        "SELECT path, role, enabled FROM project_directory_mappings WHERE project_id = ? ORDER BY path",
    )
    .bind(project_id)
    .fetch_all(&pool)
    .await?;

    let mut directory_mappings = Vec::with_capacity(mapping_rows.len());

    for row in mapping_rows {
        let role: String = row.get("role");

        directory_mappings.push(ProjectDirectoryMapping {
            path: row.get("path"),
            role: role_from_storage_value(&role)?,
            enabled: row.get("enabled"),
        });
    }

    Ok(Some(ProjectConfig {
        root_path,
        directory_mappings,
    }))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use writing_assist_core::{ProjectConfigValidationError, ProjectDirectoryMapping, ProjectDirectoryRole};

    use super::{load_project_config, save_project_config};

    fn mapping(path: &str, role: ProjectDirectoryRole, enabled: bool) -> ProjectDirectoryMapping {
        ProjectDirectoryMapping {
            path: path.to_string(),
            role,
            enabled,
        }
    }

    #[tokio::test]
    async fn saves_and_loads_project_import_configuration() {
        let project_root = tempdir().expect("project root");

        let saved = save_project_config(
            project_root.path(),
            &[
                mapping("drafts", ProjectDirectoryRole::PrimaryManuscript, true),
                mapping("lore", ProjectDirectoryRole::Reference, true),
            ],
        )
        .await
        .expect("save project config");

        let loaded = load_project_config(project_root.path())
            .await
            .expect("load project config")
            .expect("stored config");

        assert_eq!(loaded.root_path, saved.root_path);
        assert_eq!(loaded.directory_mappings, saved.directory_mappings);
    }

    #[tokio::test]
    async fn overwrites_previous_mappings_when_resaved() {
        let project_root = tempdir().expect("project root");

        save_project_config(
            project_root.path(),
            &[
                mapping("drafts", ProjectDirectoryRole::PrimaryManuscript, true),
                mapping("notes", ProjectDirectoryRole::Notes, true),
            ],
        )
        .await
        .expect("initial save");

        save_project_config(
            project_root.path(),
            &[
                mapping("chapters", ProjectDirectoryRole::PrimaryManuscript, true),
                mapping("world", ProjectDirectoryRole::Reference, true),
            ],
        )
        .await
        .expect("second save");

        let loaded = load_project_config(project_root.path())
            .await
            .expect("load project config")
            .expect("stored config");

        assert_eq!(
            loaded.directory_mappings,
            vec![
                mapping("chapters", ProjectDirectoryRole::PrimaryManuscript, true),
                mapping("world", ProjectDirectoryRole::Reference, true),
            ]
        );
    }

    #[tokio::test]
    async fn rejects_invalid_mappings() {
        let project_root = tempdir().expect("project root");

        let error = save_project_config(
            project_root.path(),
            &[mapping("notes", ProjectDirectoryRole::Notes, true)],
        )
        .await
        .expect_err("invalid config should fail");

        assert!(matches!(
            error,
            super::StoreError::Validation(
                ProjectConfigValidationError::InvalidPrimaryManuscriptCount
            )
        ));

        assert_eq!(
            load_project_config(project_root.path())
                .await
                .expect("load project config"),
            None
        );
    }
}
