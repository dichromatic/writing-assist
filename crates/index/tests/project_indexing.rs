use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use writing_assist_core::{DocumentType, ProjectDirectoryMapping, ProjectDirectoryRole};
use writing_assist_index::{classify_document_path, discover_project_documents};

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();

    let dir = std::env::temp_dir().join(format!("writing-assist-index-tests-{nanos}"));
    fs::create_dir_all(&dir).expect("temp test dir should be created");
    dir
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent directories should be created");
    }

    fs::write(path, contents).expect("test file should be written");
}

fn mapping(path: &str, role: ProjectDirectoryRole) -> ProjectDirectoryMapping {
    ProjectDirectoryMapping {
        path: path.to_string(),
        role,
        enabled: true,
    }
}

#[test]
fn classifies_markdown_files_from_configured_directory_mappings() {
    let root = PathBuf::from("/tmp/project");
    let mappings = vec![
        mapping("drafts", ProjectDirectoryRole::PrimaryManuscript),
        mapping("lore", ProjectDirectoryRole::Reference),
        mapping("notes", ProjectDirectoryRole::Notes),
    ];

    assert_eq!(
        classify_document_path(&root.join("drafts/chapter 1.md"), &root, &mappings),
        Some(DocumentType::Manuscript)
    );
    assert_eq!(
        classify_document_path(&root.join("lore/history.md"), &root, &mappings),
        Some(DocumentType::Reference)
    );
    assert_eq!(
        classify_document_path(&root.join("notes/brainstorm.md"), &root, &mappings),
        Some(DocumentType::Note)
    );
}

#[test]
fn ignores_unmapped_disabled_or_non_markdown_files() {
    let root = PathBuf::from("/tmp/project");
    let mappings = vec![
        mapping("drafts", ProjectDirectoryRole::PrimaryManuscript),
        ProjectDirectoryMapping {
            path: "archive".to_string(),
            role: ProjectDirectoryRole::Reference,
            enabled: false,
        },
        mapping("ignore-me", ProjectDirectoryRole::Ignore),
    ];

    assert_eq!(
        classify_document_path(&root.join("drafts/chapter 1.txt"), &root, &mappings),
        None
    );
    assert_eq!(
        classify_document_path(&root.join("research/history.md"), &root, &mappings),
        None
    );
    assert_eq!(
        classify_document_path(&root.join("archive/history.md"), &root, &mappings),
        None
    );
    assert_eq!(
        classify_document_path(&root.join("ignore-me/history.md"), &root, &mappings),
        None
    );
}

#[test]
fn discovers_only_markdown_files_from_enabled_mapped_directories() {
    let root = unique_temp_dir();

    write_file(&root.join("drafts/part-1/chapter 1.md"), "# Chapter 1");
    write_file(&root.join("drafts/part-1/chapter 2.md"), "# Chapter 2");
    write_file(&root.join("lore/history.md"), "# History");
    write_file(&root.join("notes/brainstorm.md"), "# Brainstorm");
    write_file(&root.join("notes/scratch.txt"), "ignored");
    write_file(&root.join("research/freeform.md"), "ignored");
    write_file(&root.join("archive/old.md"), "ignored");

    let mappings = vec![
        mapping("drafts", ProjectDirectoryRole::PrimaryManuscript),
        mapping("lore", ProjectDirectoryRole::Reference),
        mapping("notes", ProjectDirectoryRole::Notes),
        ProjectDirectoryMapping {
            path: "archive".to_string(),
            role: ProjectDirectoryRole::Reference,
            enabled: false,
        },
    ];

    let documents =
        discover_project_documents(&root, &mappings).expect("project discovery should succeed");

    let relative_paths: Vec<_> = documents
        .iter()
        .map(|document| {
            document
                .path
                .strip_prefix(&root)
                .expect("document path should be under root")
                .to_string_lossy()
                .to_string()
        })
        .collect();

    assert_eq!(
        relative_paths,
        vec![
            "drafts/part-1/chapter 1.md".to_string(),
            "drafts/part-1/chapter 2.md".to_string(),
            "lore/history.md".to_string(),
            "notes/brainstorm.md".to_string(),
        ]
    );

    assert_eq!(documents[0].document_type, DocumentType::Manuscript);
    assert_eq!(documents[1].document_type, DocumentType::Manuscript);
    assert_eq!(documents[2].document_type, DocumentType::Reference);
    assert_eq!(documents[3].document_type, DocumentType::Note);

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}
