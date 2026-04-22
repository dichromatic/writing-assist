mod bootstrapped_lexicon;
mod discovery;
mod document_archetypes;
mod document_lexicon_bootstrap;
mod entity_extraction;
mod evidence_clustering;
mod evidence_harvesting;
mod evidence_promotion;
mod exact_phrase_lexicon_matcher;
mod fact_extraction;
mod import_candidates;
mod markdown_parser;
mod project_files;
mod summary_generation;
mod text_preprocessing;

pub use bootstrapped_lexicon::induce_bootstrapped_lexicon_entries;
pub use discovery::{DiscoveredDocument, classify_document_path, discover_project_documents};
pub use document_archetypes::classify_document_archetype;
pub use document_lexicon_bootstrap::{
    DocumentLexiconBootstrapPass, IteratedDocumentLexiconBootstrap,
    iterate_document_lexicon_bootstrap,
};
pub use entity_extraction::extract_entity_candidates;
pub use evidence_clustering::cluster_document_mentions;
pub use evidence_harvesting::{
    harvest_definition_candidates, harvest_mention_candidates, harvest_section_summary_seeds,
    harvest_structured_field_candidates,
};
pub use evidence_promotion::promote_evidence_for_context;
pub use exact_phrase_lexicon_matcher::{
    CompiledExactPhraseLexiconMatcher, compile_exact_phrase_lexicon_matcher,
    harvest_exact_phrase_lexicon_mentions,
};
pub use fact_extraction::extract_reviewable_facts;
pub use import_candidates::discover_project_import_candidates;
pub use markdown_parser::{
    parse_markdown_document, parse_markdown_document_with_options, supported_span_types,
};
pub use summary_generation::generate_reviewable_summaries;
pub use text_preprocessing::preprocess_parsed_document;
