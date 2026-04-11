use std::collections::HashMap;

use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, EvidenceContext, MemorySourceReference,
    MentionCandidate, MentionFeature, ParsedMarkdownDocument, ParsedSection, ParsedSpan,
    SectionSummarySeed, SpanType, StructuredFieldCandidate, TargetAnchor,
};

const MAX_MENTION_WORDS: usize = 5;
const SUMMARY_TEXT_LIMIT: usize = 240;

#[derive(Debug, Clone)]
struct MentionObservation {
    surface: String,
    normalized_surface: String,
    source: MemorySourceReference,
    contexts: Vec<EvidenceContext>,
    occurrence_count: usize,
    features: Vec<MentionFeature>,
}

#[derive(Debug, Clone)]
struct TokenObservation {
    text: String,
    had_possessive: bool,
    ends_phrase: bool,
}

/// Harvest deterministic mention evidence without claiming semantic truth.
///
/// This phase deliberately stays below final entity extraction. It preserves
/// repeated/titled surface forms with source anchors and local context so a
/// later provider-backed pass can type, merge, or reject them.
pub fn harvest_mention_candidates(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
) -> Vec<MentionCandidate> {
    let document_path = document_path.as_ref();
    let mut observations = Vec::<MentionObservation>::new();
    let mut index_by_normalized_surface = HashMap::<String, usize>::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| matches!(span.span_type, SpanType::Heading | SpanType::Paragraph))
    {
        for harvested in mentions_in_span(document_path, span, parsed, &archetype) {
            if let Some(existing_index) =
                index_by_normalized_surface.get(&harvested.normalized_surface).copied()
            {
                let existing = &mut observations[existing_index];
                existing.occurrence_count += 1;
                merge_anchors(&mut existing.source.anchors, &harvested.source.anchors);
                merge_contexts(&mut existing.contexts, &harvested.contexts);
                merge_features(&mut existing.features, &harvested.features);
            } else {
                index_by_normalized_surface
                    .insert(harvested.normalized_surface.clone(), observations.len());
                observations.push(harvested);
            }
        }
    }

    observations
        .into_iter()
        .filter(|observation| mention_survives_aggregation(observation, &archetype))
        .map(|observation| {
            let mut features = observation.features;
            if observation.occurrence_count > 1 && !features.contains(&MentionFeature::Repeated) {
                features.push(MentionFeature::Repeated);
            }

            MentionCandidate {
                id: stable_hash_id(
                    document_path,
                    "mention",
                    &observation.normalized_surface,
                    &observation.surface,
                ),
                surface: observation.surface,
                normalized_surface: observation.normalized_surface,
                occurrence_count: observation.occurrence_count,
                source: observation.source,
                contexts: observation.contexts,
                features,
                archetype: archetype.clone(),
            }
        })
        .collect()
}

/// Harvest conservative labeled fields from structured notes.
pub fn harvest_structured_field_candidates(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
) -> Vec<StructuredFieldCandidate> {
    if matches!(archetype, DocumentArchetype::Manuscript) {
        return Vec::new();
    }

    let document_path = document_path.as_ref();
    let mut fields = Vec::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Paragraph)
    {
        for line in span.text.lines() {
            let Some((label, value)) = parse_structured_field_line(line) else {
                continue;
            };

            fields.push(StructuredFieldCandidate {
                id: stable_hash_id(document_path, "field", &label, &value),
                label,
                value,
                source: MemorySourceReference::new(
                    document_path,
                    vec![TargetAnchor::span(span.ordinal)],
                    span.start_char,
                    span.end_char,
                ),
                contexts: vec![build_context(span, parsed)],
                archetype: archetype.clone(),
            });
        }
    }

    fields
}

/// Harvest definition-like evidence from taxonomy-style references.
pub fn harvest_definition_candidates(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
) -> Vec<DefinitionCandidate> {
    if !matches!(
        archetype,
        DocumentArchetype::TaxonomyReference | DocumentArchetype::ExpositoryWorldArticle
    ) {
        return Vec::new();
    }

    let document_path = document_path.as_ref();
    let mut definitions = Vec::new();

    for span in parsed
        .spans
        .iter()
        .filter(|span| span.span_type == SpanType::Paragraph)
    {
        for line in span.text.lines() {
            let Some((term, definition)) = parse_definition_line(line) else {
                continue;
            };

            definitions.push(DefinitionCandidate {
                id: stable_hash_id(document_path, "definition", &term, &definition),
                term,
                definition,
                source: MemorySourceReference::new(
                    document_path,
                    vec![TargetAnchor::span(span.ordinal)],
                    span.start_char,
                    span.end_char,
                ),
                contexts: vec![build_context(span, parsed)],
                archetype: archetype.clone(),
            });
        }
    }

    definitions
}

/// Harvest bounded extractive seeds rather than final semantic summaries.
pub fn harvest_section_summary_seeds(
    document_path: impl AsRef<str>,
    archetype: DocumentArchetype,
    parsed: &ParsedMarkdownDocument,
) -> Vec<SectionSummarySeed> {
    let document_path = document_path.as_ref();

    parsed
        .sections
        .iter()
        .filter_map(|section| {
            let text = summarize_extractively(&section.text)?;
            let section_anchor = TargetAnchor::section(section.ordinal);

            Some(SectionSummarySeed {
                id: stable_hash_id(
                    document_path,
                    "section_summary_seed",
                    &section.ordinal.to_string(),
                    &text,
                ),
                scope: format!("section:{}", section.ordinal),
                text,
                source: MemorySourceReference::new(
                    document_path,
                    vec![section_anchor.clone()],
                    section.start_char,
                    section.end_char,
                ),
                contexts: vec![EvidenceContext {
                    span_anchor: TargetAnchor::section(section.ordinal),
                    section_anchor: Some(section_anchor),
                    heading: section_heading(section),
                    excerpt: summarize_extractively(&section.text).unwrap_or_default(),
                }],
                archetype: archetype.clone(),
            })
        })
        .collect()
}

fn mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    let tokens = span
        .normalized_text
        .split_whitespace()
        .map(clean_entity_token)
        .filter(|token| !token.text.is_empty())
        .collect::<Vec<_>>();
    let mut mentions = Vec::new();
    let mut index = 0;

    while index < tokens.len() {
        if !is_mention_token(&tokens[index].text) {
            index += 1;
            continue;
        }

        let start_index = index;
        index += 1;

        while index < tokens.len()
            && is_mention_token(&tokens[index].text)
            && !tokens[index - 1].ends_phrase
        {
            index += 1;
        }

        let mut words = tokens[start_index..index].to_vec();

        while words.len() > 1 && is_leading_drop_token(&words[0].text) {
            words.remove(0);
        }
        while words.len() > 1 && is_trailing_drop_token(&words[words.len() - 1].text) {
            words.pop();
        }

        if words.is_empty() || words.len() > MAX_MENTION_WORDS {
            continue;
        }

        let surface = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let normalized_surface = normalize_mention_surface(&surface);
        let word_count = words.len();
        let titled = words
            .first()
            .map(|first| is_title_prefix(&first.text))
            .unwrap_or(false);

        if normalized_surface.is_empty()
            || should_reject_harvested_mention(
                &surface,
                &normalized_surface,
                word_count,
                titled,
                archetype,
            )
        {
            continue;
        }

        let mut features = Vec::new();
        if word_count > 1 {
            features.push(MentionFeature::MultiWord);
        }
        if titled {
            features.push(MentionFeature::Titled);
        }
        if span.span_type == SpanType::Heading {
            features.push(MentionFeature::HeadingMentioned);
        }
        if words.iter().any(|word| word.had_possessive) {
            features.push(MentionFeature::PossessiveObserved);
        }

        mentions.push(MentionObservation {
            surface,
            normalized_surface,
            source: MemorySourceReference::new(
                document_path,
                vec![TargetAnchor::span(span.ordinal)],
                span.start_char,
                span.end_char,
            ),
            contexts: vec![build_context(span, parsed)],
            occurrence_count: 1,
            features,
        });
    }

    mentions
}

fn build_context(span: &ParsedSpan, parsed: &ParsedMarkdownDocument) -> EvidenceContext {
    let section = parsed.sections.iter().find(|section| {
        span.start_char >= section.start_char && span.end_char <= section.end_char
    });

    EvidenceContext {
        span_anchor: TargetAnchor::span(span.ordinal),
        section_anchor: section.map(|section| TargetAnchor::section(section.ordinal)),
        heading: section.and_then(section_heading),
        excerpt: truncate_to_char_limit(&span.normalized_text, SUMMARY_TEXT_LIMIT),
    }
}

fn section_heading(section: &ParsedSection) -> Option<String> {
    section
        .boundary_text
        .as_ref()
        .map(|text| text.trim_start_matches('#').trim().to_string())
        .filter(|text| !text.is_empty())
}

fn clean_entity_token(token: &str) -> TokenObservation {
    let ends_phrase = token.ends_with(',')
        || token.ends_with(';')
        || token.ends_with(':')
        || token.ends_with('—')
        || token.ends_with('–');
    let cleaned = token.trim_matches(|character: char| {
        character.is_ascii_punctuation()
            || matches!(character, '“' | '”' | '‘' | '’' | '—' | '–' | '…')
    });
    let (text, had_possessive) = if let Some(stripped) = cleaned.strip_suffix("'s") {
        (stripped, true)
    } else if let Some(stripped) = cleaned.strip_suffix("’s") {
        (stripped, true)
    } else {
        (cleaned, false)
    };

    TokenObservation {
        text: text.to_string(),
        had_possessive,
        ends_phrase,
    }
}

fn is_mention_token(token: &str) -> bool {
    let Some(first_character) = token.chars().next() else {
        return false;
    };

    first_character.is_uppercase()
        && token
            .chars()
            .any(|character| character.is_alphabetic())
}

fn is_leading_drop_token(token: &str) -> bool {
    matches!(
        token,
        "The" | "A" | "An" | "Hey" | "Oh" | "Ah" | "Well" | "Yes" | "No" | "Please"
            | "But" | "And" | "So" | "Though" | "When" | "While" | "After" | "Before"
    )
}

fn is_trailing_drop_token(token: &str) -> bool {
    is_noise_singleton(token)
}

fn is_noise_singleton(token: &str) -> bool {
    matches!(
        token,
        "The"
            | "A"
            | "An"
            | "I"
            | "We"
            | "It"
            | "He"
            | "She"
            | "They"
            | "You"
            | "My"
            | "Your"
            | "Our"
            | "Their"
            | "His"
            | "Her"
            | "This"
            | "That"
            | "These"
            | "Those"
            | "There"
            | "Here"
            | "Hey"
            | "Oh"
            | "Ah"
            | "Yes"
            | "No"
            | "Nah"
            | "Please"
            | "And"
            | "But"
            | "So"
            | "Though"
            | "When"
            | "While"
            | "After"
            | "Before"
            | "If"
            | "Then"
            | "Today"
            | "Opening"
            | "Another"
            | "Even"
            | "Just"
            | "Like"
            | "Each"
            | "Some"
            | "Any"
            | "Are"
            | "Is"
            | "As"
            | "At"
            | "On"
            | "In"
            | "Of"
            | "For"
            | "From"
            | "With"
            | "Without"
            | "Through"
            | "Because"
            | "Would"
            | "Could"
            | "Should"
            | "Has"
            | "Had"
            | "Welcome"
            | "Good"
            | "Real"
            | "Anything"
            | "Something"
            | "Everyone"
            | "Nothing"
            | "I’m"
            | "I'll"
            | "I’ll"
            | "It’s"
            | "It's"
            | "We’re"
            | "We're"
            | "We’ve"
            | "We've"
            | "That’s"
            | "That's"
            | "You’re"
            | "You're"
            | "Don’t"
            | "Don't"
            | "I’ve"
            | "I've"
    )
}

fn mention_survives_aggregation(
    observation: &MentionObservation,
    archetype: &DocumentArchetype,
) -> bool {
    if observation
        .normalized_surface
        .split_whitespace()
        .all(is_noise_singleton)
    {
        return false;
    }

    match archetype {
        DocumentArchetype::Manuscript => {
            observation.occurrence_count > 1
                || observation
                    .features
                    .iter()
                    .any(|feature| {
                        matches!(feature, MentionFeature::MultiWord | MentionFeature::Titled)
                    })
        }
        _ => true,
    }
}

fn should_reject_harvested_mention(
    surface: &str,
    normalized_surface: &str,
    word_count: usize,
    titled: bool,
    archetype: &DocumentArchetype,
) -> bool {
    if word_count == 0 {
        return true;
    }

    if word_count == 1 && is_noise_singleton(surface) {
        return true;
    }

    if normalized_surface.split_whitespace().all(is_noise_singleton) {
        return true;
    }

    match archetype {
        DocumentArchetype::Manuscript => !(word_count > 1 || titled || !is_noise_singleton(surface)),
        _ => false,
    }
}

fn is_title_prefix(token: &str) -> bool {
    matches!(
        token,
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
    )
}

fn normalize_mention_surface(surface: &str) -> String {
    surface
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_structured_field_line(line: &str) -> Option<(String, String)> {
    let trimmed = line
        .trim()
        .trim_start_matches("- ")
        .trim_start_matches("* ")
        .trim_start_matches("+ ")
        .trim();

    let (label, value) = if let Some((label, value)) = trimmed.split_once(':') {
        (label, value)
    } else if let Some((label, value)) = trimmed.split_once(" - ") {
        (label, value)
    } else if let Some((label, value)) = trimmed.split_once(" — ") {
        (label, value)
    } else {
        return None;
    };

    let label = normalize_field_part(label);
    let value = normalize_field_part(value);
    let label_word_count = label.split_whitespace().count();

    if label.is_empty()
        || value.is_empty()
        || !(1..=4).contains(&label_word_count)
        || !label.chars().any(|character| character.is_alphanumeric())
        || !value.chars().any(|character| character.is_alphanumeric())
    {
        return None;
    }

    Some((label, value))
}

fn parse_definition_line(line: &str) -> Option<(String, String)> {
    let trimmed = line
        .trim()
        .trim_start_matches("- ")
        .trim_start_matches("* ")
        .trim_start_matches("+ ")
        .trim();

    let (term, definition) = if let Some((term, definition)) = trimmed.split_once(" = ") {
        (term, definition)
    } else if let Some((term, definition)) = trimmed.split_once(':') {
        (term, definition)
    } else if let Some((term, definition)) = trimmed.split_once(" — ") {
        (term, definition)
    } else {
        return None;
    };

    let term = normalize_field_part(term);
    let definition = normalize_field_part(definition);

    if term.is_empty()
        || definition.is_empty()
        || term.split_whitespace().count() > 6
        || !term.chars().any(|character| character.is_alphanumeric())
        || !definition.chars().any(|character| character.is_alphanumeric())
    {
        return None;
    }

    Some((term, definition))
}

fn normalize_field_part(text: &str) -> String {
    text.trim()
        .trim_matches(|character: char| matches!(character, ':' | '-' | '—'))
        .trim()
        .trim_end_matches('.')
        .trim()
        .to_string()
}

fn summarize_extractively(text: &str) -> Option<String> {
    let mut summary_parts = Vec::new();
    let mut current_length = 0;

    for line in text.lines() {
        let line = line.trim();

        if line.is_empty() || matches!(line, "---" | "***" | "___") {
            continue;
        }

        let cleaned_line = line
            .trim_start_matches('#')
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        if cleaned_line.is_empty() {
            continue;
        }

        let separator_length = if summary_parts.is_empty() { 0 } else { 1 };
        let remaining = SUMMARY_TEXT_LIMIT.saturating_sub(current_length + separator_length);

        if remaining == 0 {
            break;
        }

        let snippet = truncate_to_char_limit(&cleaned_line, remaining);

        if snippet.is_empty() {
            break;
        }

        current_length += separator_length + snippet.chars().count();
        summary_parts.push(snippet);

        if current_length >= SUMMARY_TEXT_LIMIT {
            break;
        }
    }

    if summary_parts.is_empty() {
        None
    } else {
        Some(summary_parts.join(" "))
    }
}

fn truncate_to_char_limit(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

fn stable_hash_id(document_path: &str, kind: &str, left: &str, right: &str) -> Uuid {
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

    Uuid::from_u128(hash)
}

fn merge_anchors(existing: &mut Vec<TargetAnchor>, incoming: &[TargetAnchor]) {
    for anchor in incoming {
        if !existing.contains(anchor) {
            existing.push(anchor.clone());
        }
    }
}

fn merge_contexts(existing: &mut Vec<EvidenceContext>, incoming: &[EvidenceContext]) {
    for context in incoming {
        if existing.iter().any(|existing_context| existing_context == context) {
            continue;
        }

        if existing.len() >= 3 {
            break;
        }

        existing.push(context.clone());
    }
}

fn merge_features(existing: &mut Vec<MentionFeature>, incoming: &[MentionFeature]) {
    for feature in incoming {
        if !existing.contains(feature) {
            existing.push(feature.clone());
        }
    }
}
