use std::collections::HashMap;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use writing_assist_core::{
    BootstrappedLexiconEntry, DocumentArchetype, MemorySourceReference, MentionCandidate,
    MentionFeature, MentionOccurrence, ParsedMarkdownDocument, PreprocessedSentence, SpanType,
    TargetAnchor,
};

use crate::preprocess_parsed_document;

/// Compile an exact-phrase matcher backed by `aho-corasick` over normalized
/// bootstrapped lexicon surfaces.
#[derive(Debug, Clone)]
pub struct CompiledExactPhraseLexiconMatcher {
    automaton: AhoCorasick,
    patterns: Vec<CompiledLexiconPattern>,
}

#[derive(Debug, Clone)]
struct CompiledLexiconPattern {
    canonical_surface: String,
    normalized_pattern: String,
}

#[derive(Debug, Clone)]
struct LexiconSupportedObservation {
    surface: String,
    normalized_surface: String,
    source: MemorySourceReference,
    occurrences: Vec<MentionOccurrence>,
    aggregate_features: Vec<MentionFeature>,
}

pub fn compile_exact_phrase_lexicon_matcher(
    entries: &[BootstrappedLexiconEntry],
) -> CompiledExactPhraseLexiconMatcher {
    let mut patterns = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for entry in entries {
        if entry.normalized_surface.is_empty()
            || !entry
                .normalized_surface
                .chars()
                .any(|character| character.is_alphabetic())
        {
            continue;
        }

        let normalized_pattern = entry.normalized_surface.to_lowercase();
        if !seen.insert(normalized_pattern.clone()) {
            continue;
        }

        patterns.push(CompiledLexiconPattern {
            canonical_surface: entry.canonical_surface.clone(),
            normalized_pattern,
        });
    }

    let automaton = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(
            patterns
                .iter()
                .map(|pattern| pattern.normalized_pattern.as_str()),
        )
        .expect("valid bootstrapped lexicon patterns");

    CompiledExactPhraseLexiconMatcher {
        automaton,
        patterns,
    }
}

pub fn harvest_exact_phrase_lexicon_mentions(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
    matcher: &CompiledExactPhraseLexiconMatcher,
) -> Vec<MentionCandidate> {
    let document_path = document_path.as_ref();
    let preprocessed = preprocess_parsed_document(parsed);
    let mut observations = HashMap::<String, LexiconSupportedObservation>::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| matches!(span.span_type, SpanType::Heading | SpanType::Paragraph))
    {
        let Some(preprocessed_span) = preprocessed
            .spans
            .iter()
            .find(|preprocessed_span| preprocessed_span.span_ordinal == span.ordinal)
        else {
            continue;
        };

        let lowercase_haystack = preprocessed_span.normalized_text.to_lowercase();

        for matched in matcher.automaton.find_iter(&lowercase_haystack) {
            if !is_word_bounded_match(&lowercase_haystack, matched.start(), matched.end()) {
                continue;
            }

            let pattern = &matcher.patterns[matched.pattern().as_usize()];
            let matched_surface =
                slice_by_byte_range(&lowercase_haystack, matched.start(), matched.end());
            let sentence = supporting_sentence(
                preprocessed_span.sentences.as_slice(),
                &pattern.normalized_pattern,
            );
            let occurrence = build_occurrence(span, parsed, sentence, &matched_surface);
            let normalized_surface = matched_surface.clone();
            let source = MemorySourceReference::new(
                document_path,
                vec![TargetAnchor::span(span.ordinal)],
                span.start_char,
                span.end_char,
            );
            let aggregate_features =
                aggregate_features_for_surface(&matched_surface, &pattern.canonical_surface);

            observations
                .entry(normalized_surface.clone())
                .and_modify(|existing| {
                    merge_anchors(&mut existing.source.anchors, &source.anchors);
                    if !existing.occurrences.contains(&occurrence) {
                        existing.occurrences.push(occurrence.clone());
                    }
                    merge_features(&mut existing.aggregate_features, &aggregate_features);
                })
                .or_insert(LexiconSupportedObservation {
                    surface: matched_surface,
                    normalized_surface,
                    source,
                    occurrences: vec![occurrence],
                    aggregate_features,
                });
        }
    }

    observations
        .into_values()
        .map(|observation| MentionCandidate {
            id: stable_hash_id(
                document_path,
                "lexicon_supported_mention",
                &observation.normalized_surface,
                &observation.surface,
            ),
            surface: observation.surface,
            normalized_surface: observation.normalized_surface,
            source: observation.source,
            occurrences: observation.occurrences,
            aggregate_features: observation.aggregate_features,
            archetype: archetype.clone(),
        })
        .collect()
}

fn supporting_sentence<'a>(
    sentences: &'a [PreprocessedSentence],
    normalized_pattern: &str,
) -> &'a PreprocessedSentence {
    sentences
        .iter()
        .find(|sentence| {
            sentence
                .normalized_text
                .to_lowercase()
                .contains(normalized_pattern)
        })
        .unwrap_or_else(|| {
            sentences
                .first()
                .expect("preprocessed spans should preserve at least one sentence")
        })
}

fn build_occurrence(
    span: &writing_assist_core::ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    sentence: &PreprocessedSentence,
    _surface: &str,
) -> MentionOccurrence {
    let section = parsed
        .sections
        .iter()
        .find(|section| span.start_char >= section.start_char && span.end_char <= section.end_char);

    MentionOccurrence {
        span_anchor: TargetAnchor::span(span.ordinal),
        section_anchor: section.map(|section| TargetAnchor::section(section.ordinal)),
        heading: section
            .and_then(|section| section.boundary_text.as_ref())
            .map(|text| text.trim_start_matches('#').trim().to_string())
            .filter(|text| !text.is_empty()),
        snippet: sentence.normalized_text.clone(),
        sentence_type: sentence.sentence_type.clone(),
        cooccurring_mentions: Vec::new(),
    }
}

fn aggregate_features_for_surface(
    matched_surface: &str,
    canonical_surface: &str,
) -> Vec<MentionFeature> {
    let mut features = Vec::new();

    if matched_surface.split_whitespace().count() > 1 {
        features.push(MentionFeature::MultiWord);
    }

    let canonical_first = canonical_surface
        .split_whitespace()
        .next()
        .unwrap_or_default();
    if matches!(
        canonical_first,
        "Captain"
            | "Admiral"
            | "Commander"
            | "Dr"
            | "Doctor"
            | "Professor"
            | "Master"
            | "Mrs"
            | "Miss"
            | "Mr"
            | "Ms"
            | "Archmage"
            | "Pioneer"
            | "General"
            | "Elder"
    ) {
        features.push(MentionFeature::Titled);
    }

    features
}

fn is_word_bounded_match(text: &str, start: usize, end: usize) -> bool {
    let previous = text[..start].chars().next_back();
    let next = text[end..].chars().next();

    previous
        .map(|character| !character.is_alphanumeric())
        .unwrap_or(true)
        && next
            .map(|character| !character.is_alphanumeric())
            .unwrap_or(true)
}

fn slice_by_byte_range(text: &str, start: usize, end: usize) -> String {
    text[start..end].to_string()
}

fn merge_anchors(existing: &mut Vec<TargetAnchor>, incoming: &[TargetAnchor]) {
    for anchor in incoming {
        if !existing.contains(anchor) {
            existing.push(anchor.clone());
        }
    }
}

fn merge_features(existing: &mut Vec<MentionFeature>, incoming: &[MentionFeature]) {
    for feature in incoming {
        if !existing.contains(feature) {
            existing.push(feature.clone());
        }
    }
}

fn stable_hash_id(document_path: &str, kind: &str, left: &str, right: &str) -> uuid::Uuid {
    let mut hash = 0xcbf29ce484222325_u128;

    for byte in document_path
        .bytes()
        .chain([0])
        .chain(kind.bytes())
        .chain([0])
        .chain(left.bytes())
        .chain([0])
        .chain(right.bytes())
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }

    uuid::Uuid::from_u128(hash)
}
