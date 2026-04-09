#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::Path;

use writing_assist_core::{
    OpenedProject, ProjectConfig, ProjectDirectoryMapping, ProjectImportCandidate,
};
use tracing_subscriber::EnvFilter;

#[tauri::command]
fn healthcheck() -> &'static str {
    // Keep one minimal command available so frontend/runtime wiring can be verified before real features exist.
    writing_assist_orchestrator::phase_zero_status()
}

#[tauri::command]
fn scan_project_import_candidates(root: String) -> Result<Vec<ProjectImportCandidate>, String> {
    let root_path = Path::new(&root);

    if !root_path.is_dir() {
        return Err("Selected project root does not exist or is not a directory.".to_string());
    }

    // Phase 1.3 bridges the import UI to the indexing layer without binding the frontend to filesystem logic.
    writing_assist_index::discover_project_import_candidates(root_path)
        .map_err(|error| format!("Failed to scan project directories: {error}"))
}

#[tauri::command]
async fn save_project_import_configuration(
    root: String,
    mappings: Vec<ProjectDirectoryMapping>,
) -> Result<ProjectConfig, String> {
    let root_path = Path::new(&root);

    if !root_path.is_dir() {
        return Err("Selected project root does not exist or is not a directory.".to_string());
    }

    writing_assist_store::save_project_config(root_path, &mappings)
        .await
        .map_err(|error| format!("Failed to save project configuration: {error}"))
}

#[tauri::command]
async fn load_project_import_configuration(root: String) -> Result<Option<ProjectConfig>, String> {
    let root_path = Path::new(&root);

    if !root_path.is_dir() {
        return Err("Selected project root does not exist or is not a directory.".to_string());
    }

    writing_assist_store::load_project_config(root_path)
        .await
        .map_err(|error| format!("Failed to load project configuration: {error}"))
}

#[tauri::command]
async fn open_configured_project(root: String) -> Result<OpenedProject, String> {
    let root_path = Path::new(&root);

    if !root_path.is_dir() {
        return Err("Selected project root does not exist or is not a directory.".to_string());
    }

    writing_assist_orchestrator::open_configured_project(root_path)
        .await
        .map_err(|error| format!("Failed to open configured project: {error}"))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            healthcheck,
            scan_project_import_candidates,
            save_project_import_configuration,
            load_project_import_configuration,
            open_configured_project
        ])
        .run(tauri::generate_context!())
        .expect("error while running writing assist desktop application");
}
