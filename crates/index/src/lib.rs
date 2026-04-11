mod discovery;
mod entity_extraction;
mod fact_extraction;
mod import_candidates;
mod markdown_parser;
mod project_files;
mod summary_generation;

pub use discovery::{classify_document_path, discover_project_documents, DiscoveredDocument};
pub use entity_extraction::extract_entity_candidates;
pub use fact_extraction::extract_reviewable_facts;
pub use import_candidates::discover_project_import_candidates;
pub use markdown_parser::{
    parse_markdown_document, parse_markdown_document_with_options, supported_span_types,
};
pub use summary_generation::generate_reviewable_summaries;
