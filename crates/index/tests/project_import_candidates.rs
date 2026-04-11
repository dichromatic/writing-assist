use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use writing_assist_core::{ProjectDirectoryRole, ProjectImportCandidate, ProjectImportSuggestionReason};
use writing_assist_index::discover_project_import_candidates;

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();

    let dir = std::env::temp_dir().join(format!("writing-assist-import-candidates-{nanos}"));
    fs::create_dir_all(&dir).expect("temp test dir should be created");
    dir
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent directories should be created");
    }

    fs::write(path, contents).expect("test file should be written");
}

fn candidate_by_path<'a>(
    candidates: &'a [ProjectImportCandidate],
    path: &str,
) -> &'a ProjectImportCandidate {
    candidates
        .iter()
        .find(|candidate| candidate.path == path)
        .unwrap_or_else(|| panic!("expected candidate for path {path}"))
}

#[test]
fn discovers_only_immediate_child_directories_in_stable_order() {
    let root = unique_temp_dir();

    fs::create_dir_all(root.join("chapters/part-1")).expect("chapters dir should exist");
    fs::create_dir_all(root.join("empty")).expect("empty dir should exist");
    fs::create_dir_all(root.join("notes")).expect("notes dir should exist");
    fs::create_dir_all(root.join("research")).expect("research dir should exist");
    write_file(&root.join("chapters/part-1/chapter-1.md"), "# Chapter 1");
    write_file(&root.join("README.md"), "# ignored root file");

    let candidates =
        discover_project_import_candidates(&root).expect("candidate discovery should succeed");

    let paths: Vec<_> = candidates.iter().map(|candidate| candidate.path.clone()).collect();

    assert_eq!(
        paths,
        vec![
            ".".to_string(),
            "chapters".to_string(),
            "empty".to_string(),
            "notes".to_string(),
            "research".to_string(),
        ]
    );

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}

#[test]
fn root_markdown_files_create_a_root_candidate() {
    let root = unique_temp_dir();

    write_file(&root.join("chapter-1.MD"), "# Chapter 1");

    let candidates =
        discover_project_import_candidates(&root).expect("candidate discovery should succeed");
    let root_candidate = candidate_by_path(&candidates, ".");

    assert!(root_candidate.contains_supported_text_files);
    assert_eq!(
        root_candidate.suggestion_reasons,
        vec![ProjectImportSuggestionReason::ContainsSupportedTextFiles]
    );

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}

#[test]
fn hidden_child_directories_are_not_import_candidates() {
    let root = unique_temp_dir();

    fs::create_dir_all(root.join(".writing-assist")).expect("app state dir should exist");
    fs::create_dir_all(root.join(".git")).expect("git dir should exist");
    fs::create_dir_all(root.join("chapters")).expect("chapters dir should exist");

    let candidates =
        discover_project_import_candidates(&root).expect("candidate discovery should succeed");
    let paths: Vec<_> = candidates.iter().map(|candidate| candidate.path.clone()).collect();

    assert_eq!(paths, vec!["chapters".to_string()]);

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}

#[test]
fn detects_supported_text_presence_recursively_and_marks_empty_directories() {
    let root = unique_temp_dir();

    fs::create_dir_all(root.join("chapters/part-1")).expect("chapters dir should exist");
    fs::create_dir_all(root.join("world_context")).expect("world_context dir should exist");
    fs::create_dir_all(root.join("empty")).expect("empty dir should exist");
    write_file(&root.join("chapters/part-1/chapter-1.md"), "# Chapter 1");
    write_file(&root.join("world_context/history.txt"), "plain text world note");

    let candidates =
        discover_project_import_candidates(&root).expect("candidate discovery should succeed");

    assert!(candidate_by_path(&candidates, "chapters").contains_supported_text_files);
    assert!(candidate_by_path(&candidates, "world_context").contains_supported_text_files);
    assert!(!candidate_by_path(&candidates, "empty").contains_supported_text_files);

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}

#[test]
fn suggests_roles_from_conservative_directory_name_heuristics() {
    let root = unique_temp_dir();

    fs::create_dir_all(root.join("chapters")).expect("chapters dir should exist");
    fs::create_dir_all(root.join("world_context")).expect("world_context dir should exist");
    fs::create_dir_all(root.join("notes")).expect("notes dir should exist");
    fs::create_dir_all(root.join("misc")).expect("misc dir should exist");
    write_file(&root.join("chapters/chapter-1.md"), "# Chapter 1");
    write_file(&root.join("world_context/history.md"), "# History");
    write_file(&root.join("misc/random.md"), "# Random");

    let candidates =
        discover_project_import_candidates(&root).expect("candidate discovery should succeed");

    let chapters = candidate_by_path(&candidates, "chapters");
    assert_eq!(chapters.suggested_role, Some(ProjectDirectoryRole::PrimaryManuscript));
    assert!(chapters
        .suggestion_reasons
        .contains(&ProjectImportSuggestionReason::DirectoryNamedChapters));
    assert!(chapters
        .suggestion_reasons
        .contains(&ProjectImportSuggestionReason::ContainsSupportedTextFiles));

    let world = candidate_by_path(&candidates, "world_context");
    assert_eq!(world.suggested_role, Some(ProjectDirectoryRole::Reference));
    assert!(world
        .suggestion_reasons
        .contains(&ProjectImportSuggestionReason::DirectoryNamedWorldContext));

    let notes = candidate_by_path(&candidates, "notes");
    assert_eq!(notes.suggested_role, Some(ProjectDirectoryRole::Notes));
    assert!(notes
        .suggestion_reasons
        .contains(&ProjectImportSuggestionReason::DirectoryNamedNotes));

    let misc = candidate_by_path(&candidates, "misc");
    assert_eq!(misc.suggested_role, None);
    assert_eq!(
        misc.suggestion_reasons,
        vec![ProjectImportSuggestionReason::ContainsSupportedTextFiles]
    );

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}
