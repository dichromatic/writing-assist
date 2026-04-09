#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::Path;

use writing_assist_core::ProjectImportCandidate;
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

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            healthcheck,
            scan_project_import_candidates
        ])
        .run(tauri::generate_context!())
        .expect("error while running writing assist desktop application");
}
