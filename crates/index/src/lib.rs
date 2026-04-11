mod discovery;
mod document_archetypes;
mod evidence_clustering;
mod evidence_harvesting;
mod entity_extraction;
mod fact_extraction;
mod import_candidates;
mod markdown_parser;
mod project_files;
mod summary_generation;

pub use discovery::{classify_document_path, discover_project_documents, DiscoveredDocument};
pub use document_archetypes::classify_document_archetype;
pub use evidence_clustering::cluster_document_mentions;
pub use evidence_harvesting::{
    harvest_definition_candidates, harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates,
};
pub use entity_extraction::extract_entity_candidates;
pub use fact_extraction::extract_reviewable_facts;
pub use import_candidates::discover_project_import_candidates;
pub use markdown_parser::{
    parse_markdown_document, parse_markdown_document_with_options, supported_span_types,
};
pub use summary_generation::generate_reviewable_summaries;
