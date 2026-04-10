use std::collections::HashMap;

use uuid::Uuid;
use writing_assist_core::{
    EntityCandidate, MemoryReviewState, MemorySourceReference, MemoryStalenessState,
    ParsedMarkdownDocument, ParsedSpan, SpanType, TargetAnchor,
};

#[derive(Debug, Clone)]
struct EntityMentionAccumulator {
    display_name: String,
    normalized_name: String,
    first_start_char: usize,
    first_end_char: usize,
    mention_count: usize,
    anchors: Vec<TargetAnchor>,
    word_count: usize,
}

/// Extract conservative, review-gated entity candidates from parsed spans.
///
/// This Phase 3.2 extractor is deliberately deterministic and local. It only
/// produces pending candidates with source anchors; persistence, review UI, and
/// LLM-backed extraction remain later phases.
pub fn extract_entity_candidates(
    document_path: impl AsRef<str>,
    parsed: &ParsedMarkdownDocument,
) -> Vec<EntityCandidate> {
    let document_path = document_path.as_ref();
    let mut index_by_normalized_name = HashMap::new();
    let mut accumulators = Vec::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| {
            span.span_type == SpanType::Heading || span.span_type == SpanType::Paragraph
        })
    {
        for mention in entity_mentions_in_span(span) {
            let normalized_name = normalize_entity_name(&mention);
            let word_count = mention.split_whitespace().count();

            if let Some(index) = index_by_normalized_name.get(&normalized_name).copied() {
                let accumulator: &mut EntityMentionAccumulator = &mut accumulators[index];
                accumulator.mention_count += 1;

                let anchor = TargetAnchor::span(span.ordinal);
                if !accumulator.anchors.contains(&anchor) {
                    accumulator.anchors.push(anchor);
                }
            } else {
                index_by_normalized_name.insert(normalized_name.clone(), accumulators.len());
                accumulators.push(EntityMentionAccumulator {
                    display_name: mention,
                    normalized_name,
                    first_start_char: span.start_char,
                    first_end_char: span.end_char,
                    mention_count: 1,
                    anchors: vec![TargetAnchor::span(span.ordinal)],
                    word_count,
                });
            }
        }
    }

    accumulators
        .into_iter()
        .filter(|candidate| candidate.word_count > 1 || candidate.mention_count > 1)
        .map(|candidate| {
            EntityCandidate::new(
                stable_entity_id(document_path, &candidate.normalized_name),
                candidate.display_name,
                MemorySourceReference::new(
                    document_path,
                    candidate.anchors,
                    candidate.first_start_char,
                    candidate.first_end_char,
                ),
                MemoryReviewState::Pending,
                MemoryStalenessState::Current,
            )
        })
        .collect()
}

fn entity_mentions_in_span(span: &ParsedSpan) -> Vec<String> {
    let cleaned_words: Vec<_> = span
        .normalized_text
        .split_whitespace()
        .map(clean_entity_token)
        .collect();
    let mut mentions = Vec::new();
    let mut index = 0;

    while index < cleaned_words.len() {
        let word = cleaned_words[index];

        if !is_entity_token(word) {
            index += 1;
            continue;
        }

        let start_index = index;
        index += 1;

        while index < cleaned_words.len() && is_entity_token(cleaned_words[index]) {
            index += 1;
        }

        let mut words = cleaned_words[start_index..index]
            .iter()
            .copied()
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>();

        // Avoid turning sentence openers such as "The" into entity names while still
        // recovering phrases like "The Radiant Firth" as "Radiant Firth".
        while words.len() > 1 && is_leading_article(words[0]) {
            words.remove(0);
        }

        if words.is_empty() || is_noise_singleton(words.as_slice()) {
            continue;
        }

        mentions.push(words.join(" "));
    }

    mentions
}

fn clean_entity_token(token: &str) -> &str {
    token.trim_matches(|character: char| {
        character.is_ascii_punctuation()
            || matches!(character, '“' | '”' | '‘' | '’' | '—' | '–' | '…')
    })
}

fn is_entity_token(token: &str) -> bool {
    let Some(first_character) = token.chars().next() else {
        return false;
    };

    first_character.is_uppercase()
        && token
            .chars()
            .any(|character| character.is_alphabetic())
}

fn is_leading_article(word: &str) -> bool {
    matches!(word, "The" | "A" | "An")
}

fn is_noise_singleton(words: &[&str]) -> bool {
    matches!(words, [word] if is_leading_article(word))
}

fn normalize_entity_name(name: &str) -> String {
    name.split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn stable_entity_id(document_path: &str, normalized_name: &str) -> Uuid {
    let mut hash = 0xcbf29ce484222325_u128;

    for byte in document_path
        .bytes()
        .chain([0])
        .chain(normalized_name.bytes())
    {
        hash ^= byte as u128;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }

    Uuid::from_u128(hash)
}
