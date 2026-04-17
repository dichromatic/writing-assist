use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use semantic_pass_lab_tool::{
    build_semantic_pass_lab_packet, execute_semantic_pass, NimLabConfig,
    SemanticBundleShape,
};
use writing_assist_core::DocumentType;
use writing_assist_index::{
    classify_document_archetype, cluster_document_mentions, harvest_definition_candidates,
    harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates, parse_markdown_document,
};

const EXAMPLE_ROOT: &str = "/workspace/examples";
const OUTPUT_ROOT: &str = "/workspace/logs/semantic-pass-lab";
const EXPERIMENT_NAME: &str = "semantic-pass-lab-v1";
const FILE_SELECTION_ENV: &str = "SEMANTIC_PASS_FILES";
const BUNDLE_SELECTION_ENV: &str = "SEMANTIC_PASS_BUNDLE_SHAPES";
const ALL_FILES_SELECTOR: &str = "all";

const REPRESENTATIVE_FILES: &[&str] = &[
    "1. Radiant Firth.md",
    "story planning/estuary crew summaries.txt",
    "world context/tau sectors.txt",
];

fn main() -> Result<()> {
    let example_root = PathBuf::from(EXAMPLE_ROOT);
    let output_root = PathBuf::from(OUTPUT_ROOT);
    let summary_log_path = output_root.join("summary.log");
    fs::create_dir_all(&output_root).expect("semantic pass lab output directory should exist");
    let live_mode_enabled = live_mode_enabled();
    let nim_config = if live_mode_enabled {
        Some(NimLabConfig::from_env()?)
    } else {
        None
    };
    let selected_files = selected_files(&example_root);
    let selected_bundle_shapes = selected_bundle_shapes();

    let mut summary_lines = Vec::new();
    summary_lines.push(format!("experiment: {EXPERIMENT_NAME}"));
    summary_lines.push(format!("example_root: {}", example_root.display()));
    summary_lines.push(format!("output_root: {}", output_root.display()));
    summary_lines.push(format!("live_mode_enabled: {live_mode_enabled}"));
    if let Some(config) = &nim_config {
        summary_lines.push(format!("nim_model: {}", config.model));
        summary_lines.push(format!("nim_base_url: {}", config.base_url));
    }
    summary_lines.push(format!("selected_files: {:?}", selected_files));
    summary_lines.push(format!("selected_bundle_shapes: {:?}", selected_bundle_shapes));
    summary_lines.push(String::new());

    for relative_path in selected_files {
        let file_path = example_root.join(&relative_path);
        let text = fs::read_to_string(&file_path).expect("representative example file should load");
        let document_type = infer_example_document_type(&relative_path);
        let parsed = parse_markdown_document(&text);
        let archetype = classify_document_archetype(
            document_type.clone(),
            &relative_path,
            &text,
            &parsed,
        );
        let mentions = harvest_mention_candidates(&relative_path, archetype.clone(), &parsed);
        let structured_fields =
            harvest_structured_field_candidates(&relative_path, archetype.clone(), &parsed);
        let definitions =
            harvest_definition_candidates(&relative_path, archetype.clone(), &parsed);
        let summary_seeds =
            harvest_section_summary_seeds(&relative_path, archetype.clone(), &parsed);
        let mention_clusters = cluster_document_mentions(
            &relative_path,
            archetype.clone(),
            &mentions,
            &structured_fields,
            &definitions,
            &summary_seeds,
        );

        for bundle_shape in selected_bundle_shapes.clone() {
            let packet = build_semantic_pass_lab_packet(
                EXPERIMENT_NAME,
                relative_path.as_str(),
                document_type.clone(),
                archetype.clone(),
                bundle_shape,
                &mention_clusters,
                &structured_fields,
                &definitions,
                &summary_seeds,
            );
            let slug = file_slug(&relative_path);
            let shape_slug = match bundle_shape {
                SemanticBundleShape::HighRecall => "high-recall",
                SemanticBundleShape::Curated => "curated",
            };
            let artifact_root = output_root.join(slug).join(shape_slug);
            fs::create_dir_all(&artifact_root).expect("artifact output directory should exist");
            let json_path = artifact_root.join("packet.json");
            let prompt_path = artifact_root.join("prompt.txt");

            fs::write(
                &json_path,
                serde_json::to_string_pretty(&packet).expect("packet should serialize"),
            )
            .expect("packet json should write");
            fs::write(&prompt_path, &packet.prompt).expect("prompt file should write");

            let mut live_response_path = None;
            if let Some(config) = &nim_config {
                let semantic_response = execute_semantic_pass(config, &packet)?;
                let response_path = artifact_root.join("response.json");
                fs::write(
                    &response_path,
                    serde_json::to_string_pretty(&semantic_response)
                        .expect("semantic response should serialize"),
                )
                .expect("semantic response should write");
                live_response_path = Some(response_path);
            }

            summary_lines.push(format!("=== {relative_path} / {shape_slug} ==="));
            summary_lines.push(format!("document_type: {:?}", document_type));
            summary_lines.push(format!("archetype: {:?}", archetype));
            summary_lines.push(format!("clusters: {}", packet.clusters.len()));
            summary_lines.push(format!(
                "structured_fields: {}",
                packet.structured_fields.len()
            ));
            summary_lines.push(format!("definitions: {}", packet.definitions.len()));
            summary_lines.push(format!(
                "section_summary_seeds: {}",
                packet.section_summary_seeds.len()
            ));
            summary_lines.push(format!("json: {}", json_path.display()));
            summary_lines.push(format!("prompt: {}", prompt_path.display()));
            if let Some(response_path) = live_response_path {
                summary_lines.push(format!("response: {}", response_path.display()));
            }
            summary_lines.push("cluster_surfaces:".to_string());
            for cluster in &packet.clusters {
                summary_lines.push(format!(
                    "- {} | features={:?} | links={}",
                    cluster.display_surface,
                    cluster.aggregate_features,
                    cluster.linked_evidence_summaries.len()
                ));
            }
            summary_lines.push(String::new());
        }
    }

    fs::write(&summary_log_path, summary_lines.join("\n"))
        .expect("semantic pass lab summary log should write");
    println!("wrote {}", summary_log_path.display());
    Ok(())
}

fn infer_example_document_type(relative_path: &str) -> DocumentType {
    if relative_path.starts_with("world context/") {
        DocumentType::Reference
    } else if relative_path.starts_with("story planning/") {
        DocumentType::Note
    } else {
        DocumentType::Manuscript
    }
}

fn live_mode_enabled() -> bool {
    matches!(
        std::env::var("NIM_RUN_LIVE").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn selected_files(example_root: &Path) -> Vec<String> {
    std::env::var(FILE_SELECTION_ENV)
        .ok()
        .map(|value| selected_files_from_env(example_root, &value))
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| REPRESENTATIVE_FILES.iter().map(|value| value.to_string()).collect())
}

fn selected_files_from_env(example_root: &Path, raw: &str) -> Vec<String> {
    let values = parse_csv_values(raw);

    if values.iter().any(|value| value.eq_ignore_ascii_case(ALL_FILES_SELECTOR)) {
        return collect_supported_example_files(example_root)
            .into_iter()
            .map(|path| {
                path.strip_prefix(example_root)
                    .expect("example path should be relative to example root")
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
    }

    values
}

fn selected_bundle_shapes() -> Vec<SemanticBundleShape> {
    std::env::var(BUNDLE_SELECTION_ENV)
        .ok()
        .map(|value| {
            parse_csv_values(&value)
                .into_iter()
                .filter_map(|value| match value.as_str() {
                    "high-recall" | "high_recall" => Some(SemanticBundleShape::HighRecall),
                    "curated" => Some(SemanticBundleShape::Curated),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            vec![
                SemanticBundleShape::HighRecall,
                SemanticBundleShape::Curated,
            ]
        })
}

fn parse_csv_values(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        artifact_directory_name, parse_csv_values, selected_bundle_shapes,
        selected_files_from_env, SemanticBundleShape, ALL_FILES_SELECTOR,
        BUNDLE_SELECTION_ENV,
    };

    #[test]
    fn parses_csv_values_without_empty_entries() {
        let values = parse_csv_values("curated, , high-recall,,");

        assert_eq!(values, vec!["curated", "high-recall"]);
    }

    #[test]
    fn bundle_shape_env_allows_single_curated_run() {
        std::env::set_var(BUNDLE_SELECTION_ENV, "curated");

        let selected = selected_bundle_shapes();
        std::env::remove_var(BUNDLE_SELECTION_ENV);

        assert_eq!(selected, vec![SemanticBundleShape::Curated]);
    }

    #[test]
    fn selected_files_env_supports_all_keyword() {
        let example_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let selected = selected_files_from_env(&example_root, ALL_FILES_SELECTOR);

        assert!(selected.iter().any(|path| path == "1. Radiant Firth.md"));
        assert!(selected.iter().any(|path| path == "world context/tau sectors.txt"));
    }

    #[test]
    fn artifact_directory_name_preserves_relative_structure_readably() {
        assert_eq!(
            artifact_directory_name("world context/tau sectors.txt"),
            "world-context--tau-sectors_txt"
        );
    }
}

#[allow(dead_code)]
fn collect_supported_example_files(example_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_supported_example_files_recursive(example_root, &mut files);
    files.sort();
    files
}

#[allow(dead_code)]
fn collect_supported_example_files_recursive(path: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(path).expect("directory should be readable");

    for entry in entries {
        let entry = entry.expect("directory entry should load");
        let entry_path = entry.path();

        if entry_path.is_dir() {
            collect_supported_example_files_recursive(&entry_path, files);
            continue;
        }

        if matches!(
            entry_path.extension().and_then(|extension| extension.to_str()),
            Some("md" | "txt")
        ) {
            files.push(entry_path);
        }
    }
}

fn artifact_directory_name(relative_path: &str) -> String {
    relative_path
        .replace(['/', '\\'], "--")
        .replace(' ', "-")
        .replace('.', "_")
        .to_lowercase()
}

fn file_slug(relative_path: &str) -> String {
    artifact_directory_name(relative_path)
}
