use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use writing_assist_core::DocumentType;
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

#[test]
fn classifies_markdown_files_in_known_project_folders() {
    let root = PathBuf::from("/tmp/project");

    assert_eq!(
        classify_document_path(&root.join("chapters/chapter 1.md"), &root),
        Some(DocumentType::Chapter)
    );
    assert_eq!(
        classify_document_path(&root.join("world_context/history.md"), &root),
        Some(DocumentType::Reference)
    );
}

#[test]
fn ignores_non_markdown_files_and_unknown_folders() {
    let root = PathBuf::from("/tmp/project");

    assert_eq!(
        classify_document_path(&root.join("chapters/notes.txt"), &root),
        None
    );
    assert_eq!(
        classify_document_path(&root.join("notes/brainstorm.md"), &root),
        None
    );
}

#[test]
fn discovers_only_supported_project_documents() {
    let root = unique_temp_dir();

    write_file(&root.join("chapters/chapter 1.md"), "# Chapter 1");
    write_file(&root.join("chapters/chapter 2.md"), "# Chapter 2");
    write_file(&root.join("world_context/history.md"), "# History");
    write_file(&root.join("world_context/glossary.txt"), "ignored");
    write_file(&root.join("notes/freeform.md"), "ignored");

    let documents = discover_project_documents(&root).expect("project discovery should succeed");

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
            "chapters/chapter 1.md".to_string(),
            "chapters/chapter 2.md".to_string(),
            "world_context/history.md".to_string()
        ]
    );

    assert_eq!(documents[0].document_type, DocumentType::Chapter);
    assert_eq!(documents[1].document_type, DocumentType::Chapter);
    assert_eq!(documents[2].document_type, DocumentType::Reference);

    fs::remove_dir_all(root).expect("temp test dir should be removed");
}
