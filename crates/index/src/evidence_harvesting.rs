use std::collections::HashMap;

use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, EvidenceContext, MemorySourceReference,
    MentionCandidate, MentionFeature, MentionOccurrence, ParsedMarkdownDocument, ParsedSection,
    ParsedSpan, SectionSummarySeed, SentenceType, SpanType, StructuredFieldCandidate,
    TargetAnchor,
};

const MAX_MENTION_WORDS: usize = 5;
const SUMMARY_TEXT_LIMIT: usize = 240;

#[derive(Debug, Clone)]
struct MentionObservation {
    surface: String,
    normalized_surface: String,
    source: MemorySourceReference,
    occurrences: Vec<MentionOccurrence>,
    aggregate_features: Vec<MentionFeature>,
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
        for harvested in mention_observations_in_span(document_path, span, parsed, &archetype) {
            if let Some(existing_index) =
                index_by_normalized_surface.get(&harvested.normalized_surface).copied()
            {
                let existing = &mut observations[existing_index];
                merge_anchors(&mut existing.source.anchors, &harvested.source.anchors);
                merge_occurrences(&mut existing.occurrences, &harvested.occurrences);
                merge_features(
                    &mut existing.aggregate_features,
                    &harvested.aggregate_features,
                );
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
            let mut aggregate_features = observation.aggregate_features;
            if observation.occurrences.len() > 1
                && !aggregate_features.contains(&MentionFeature::Repeated)
            {
                aggregate_features.push(MentionFeature::Repeated);
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
                source: observation.source,
                occurrences: observation.occurrences,
                aggregate_features,
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

fn mention_observations_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    match archetype {
        DocumentArchetype::Manuscript => {
            capitalized_mentions_in_span(document_path, span, parsed, archetype)
        }
        DocumentArchetype::DossierProfile => {
            let mut observations =
                capitalized_mentions_in_span(document_path, span, parsed, archetype);
            observations.extend(alias_field_mentions_in_span(document_path, span, parsed));
            observations
        }
        DocumentArchetype::TaxonomyReference | DocumentArchetype::ExpositoryWorldArticle => {
            let mut observations =
                capitalized_mentions_in_span(document_path, span, parsed, archetype);
            observations.extend(definition_term_mentions_in_span(
                document_path,
                span,
                parsed,
            ));
            observations
        }
        DocumentArchetype::StoryPlanning => {
            let mut observations =
                capitalized_mentions_in_span(document_path, span, parsed, archetype);
            observations.extend(story_planning_field_mentions_in_span(
                document_path,
                span,
                parsed,
            ));
            observations
        }
        DocumentArchetype::LooseNote => {
            loose_note_mentions_in_span(document_path, span, parsed, archetype)
        }
    }
}

fn loose_note_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    capitalized_mentions_in_span(document_path, span, parsed, archetype)
        .into_iter()
        .filter(|observation| !should_reject_loose_note_observation(observation, span))
        .collect()
}

fn capitalized_mentions_in_span(
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

        let mut aggregate_features = aggregate_features_for_surface(span, &surface);
        if words.iter().any(|word| word.had_possessive)
            && !aggregate_features.contains(&MentionFeature::PossessiveObserved)
        {
            aggregate_features.push(MentionFeature::PossessiveObserved);
        }

        mentions.push(build_surface_observation(
            document_path,
            span,
            parsed,
            surface,
            aggregate_features,
        ));
    }

    mentions
}

fn alias_field_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
) -> Vec<MentionObservation> {
    let mut mentions = Vec::new();

    for line in span.text.lines() {
        let Some((label, value)) = parse_structured_field_line(line) else {
            continue;
        };

        if !is_alias_like_label(&label) || !value.chars().any(|character| character.is_alphabetic()) {
            continue;
        }

        if contains_emoji_characters(&value) {
            continue;
        }

        mentions.push(build_surface_observation(
            document_path,
            span,
            parsed,
            value.clone(),
            aggregate_features_for_surface(span, &value),
        ));
    }

    mentions
}

fn definition_term_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
) -> Vec<MentionObservation> {
    let mut mentions = Vec::new();

    for line in span.text.lines() {
        let Some((term, _definition)) = parse_definition_line(line) else {
            continue;
        };

        if !term.chars().any(|character| character.is_alphabetic()) {
            continue;
        }

        if contains_emoji_characters(&term) {
            continue;
        }

        mentions.push(build_surface_observation(
            document_path,
            span,
            parsed,
            term.clone(),
            aggregate_features_for_surface(span, &term),
        ));
    }

    mentions
}

fn story_planning_field_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
) -> Vec<MentionObservation> {
    let mut mentions = Vec::new();

    for line in span.text.lines() {
        let Some((label, value)) = parse_structured_field_line(line) else {
            continue;
        };

        if !is_story_planning_participant_label(&label) {
            continue;
        }

        for surface in split_story_planning_mentions(&value) {
            if contains_emoji_characters(&surface) {
                continue;
            }

            mentions.push(build_surface_observation(
                document_path,
                span,
                parsed,
                surface.clone(),
                aggregate_features_for_surface(span, &surface),
            ));
        }
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

fn build_occurrence(
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    surface: &str,
) -> MentionOccurrence {
    let section = parsed.sections.iter().find(|section| {
        span.start_char >= section.start_char && span.end_char <= section.end_char
    });

    MentionOccurrence {
        span_anchor: TargetAnchor::span(span.ordinal),
        section_anchor: section.map(|section| TargetAnchor::section(section.ordinal)),
        heading: section.and_then(section_heading),
        snippet: build_occurrence_snippet(&span.normalized_text, surface),
        sentence_type: classify_sentence_type(span),
        cooccurring_mentions: cooccurring_mentions_in_span(span, surface),
    }
}

fn build_surface_observation(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    surface: String,
    aggregate_features: Vec<MentionFeature>,
) -> MentionObservation {
    MentionObservation {
        normalized_surface: normalize_mention_surface(&surface),
        source: MemorySourceReference::new(
            document_path,
            vec![TargetAnchor::span(span.ordinal)],
            span.start_char,
            span.end_char,
        ),
        occurrences: vec![build_occurrence(span, parsed, &surface)],
        surface,
        aggregate_features,
    }
}

fn aggregate_features_for_surface(span: &ParsedSpan, surface: &str) -> Vec<MentionFeature> {
    let mut aggregate_features = Vec::new();
    let word_count = surface.split_whitespace().count();
    let titled = surface
        .split_whitespace()
        .next()
        .map(is_title_prefix)
        .unwrap_or(false);

    if word_count > 1 {
        aggregate_features.push(MentionFeature::MultiWord);
    }
    if titled {
        aggregate_features.push(MentionFeature::Titled);
    }
    if span.span_type == SpanType::Heading {
        aggregate_features.push(MentionFeature::HeadingMentioned);
    }

    aggregate_features
}

fn section_heading(section: &ParsedSection) -> Option<String> {
    section
        .boundary_text
        .as_ref()
        .map(|text| text.trim_start_matches('#').trim().to_string())
        .filter(|text| !text.is_empty())
}

fn clean_entity_token(token: &str) -> TokenObservation {
    let token_for_phrase_break = token.trim_end_matches(|character: char| {
        matches!(character, '“' | '”' | '‘' | '’' | '"' | '\'')
    });
    let abbreviation_without_punctuation = token_for_phrase_break
        .trim_matches(|character: char| {
            character.is_ascii_punctuation()
                || matches!(character, '“' | '”' | '‘' | '’' | '—' | '–' | '…')
        });
    let period_ends_phrase = token_for_phrase_break.ends_with('.')
        && !is_title_prefix(abbreviation_without_punctuation);
    let ends_phrase = token_for_phrase_break.ends_with(',')
        || token_for_phrase_break.ends_with(';')
        || token_for_phrase_break.ends_with(':')
        || period_ends_phrase
        || token_for_phrase_break.ends_with('!')
        || token_for_phrase_break.ends_with('?')
        || token_for_phrase_break.ends_with('—')
        || token_for_phrase_break.ends_with('–');
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

fn classify_sentence_type(span: &ParsedSpan) -> SentenceType {
    let trimmed = span.text.trim_start();

    if span.span_type == SpanType::Heading {
        return SentenceType::Heading;
    }

    if trimmed.starts_with('>') {
        return SentenceType::BlockQuote;
    }

    if trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || starts_with_numbered_list_item(trimmed)
    {
        return SentenceType::ListItem;
    }

    if trimmed.starts_with('"')
        || trimmed.starts_with('“')
        || trimmed.starts_with('‘')
        || trimmed.starts_with('\'')
        || trimmed.starts_with('—')
    {
        return SentenceType::Dialogue;
    }

    SentenceType::Narrative
}

fn starts_with_numbered_list_item(text: &str) -> bool {
    let mut characters = text.chars();
    let mut saw_digit = false;

    while let Some(character) = characters.next() {
        if character.is_ascii_digit() {
            saw_digit = true;
            continue;
        }

        return saw_digit && character == '.';
    }

    false
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
            | "As" | "How" | "Since" | "Wait"
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
            | "Wait"
            | "If"
            | "Since"
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
            | "Did"
            | "Has"
            | "Had"
            | "What"
            | "Let"
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
            | "I-I"
    )
}

fn contains_emoji_characters(text: &str) -> bool {
    text.chars().any(is_emoji_character)
}

fn is_emoji_character(character: char) -> bool {
    // Reject emoji-bearing surfaces early so decorative glyphs do not get
    // promoted into mention evidence or later semantic inputs.
    let scalar = character as u32;

    matches!(scalar, 0x200D | 0xFE0F | 0x20E3)
        || (0x1F1E6..=0x1F1FF).contains(&scalar)
        || (0x1F300..=0x1F5FF).contains(&scalar)
        || (0x1F600..=0x1F64F).contains(&scalar)
        || (0x1F680..=0x1F6FF).contains(&scalar)
        || (0x1F700..=0x1F77F).contains(&scalar)
        || (0x1F780..=0x1F7FF).contains(&scalar)
        || (0x1F800..=0x1F8FF).contains(&scalar)
        || (0x1F900..=0x1F9FF).contains(&scalar)
        || (0x1FA70..=0x1FAFF).contains(&scalar)
        || (0x2600..=0x26FF).contains(&scalar)
        || (0x2700..=0x27BF).contains(&scalar)
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
            observation.occurrences.len() > 1
                || observation
                    .aggregate_features
                    .iter()
                    .any(|feature| {
                        matches!(feature, MentionFeature::MultiWord | MentionFeature::Titled)
                    })
        }
        DocumentArchetype::LooseNote => {
            if observation.surface.split_whitespace().count() == 1
                && observation
                    .occurrences
                    .iter()
                    .all(|occurrence| occurrence.sentence_type == SentenceType::ListItem)
                && !observation.aggregate_features.iter().any(|feature| {
                    matches!(
                        feature,
                        MentionFeature::MultiWord
                            | MentionFeature::Titled
                            | MentionFeature::PossessiveObserved
                    )
                })
                && observation.occurrences.iter().all(|occurrence| {
                    occurrence
                        .snippet
                        .trim_start_matches("- ")
                        .starts_with(&observation.surface)
                })
            {
                return false;
            }

            true
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

    if contains_emoji_characters(surface) {
        return true;
    }

    if normalized_surface.split_whitespace().all(is_noise_singleton) {
        return true;
    }

    match archetype {
        DocumentArchetype::Manuscript => {
            if word_count > 1
                && surface
                    .split_whitespace()
                    .next()
                    .map(is_leading_drop_token)
                    .unwrap_or(false)
            {
                return true;
            }

            !(word_count > 1 || titled || !is_noise_singleton(surface))
        }
        _ => false,
    }
}

fn should_reject_loose_note_observation(
    observation: &MentionObservation,
    span: &ParsedSpan,
) -> bool {
    let surface_word_count = observation.surface.split_whitespace().count();
    let lower_surface = observation.surface.to_lowercase();

    for line in span.text.lines() {
        let normalized_line = line
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let lower_line = normalized_line.to_lowercase();

        if surface_word_count == 1
            && lower_line.starts_with(&format!("{lower_surface} "))
            && follows_loose_note_label_pattern(lower_line[lower_surface.len()..].trim_start())
        {
            return true;
        }

        if let Some((label, value)) = parse_structured_field_line(line.trim()) {
            let normalized_label = normalize_mention_surface(&label);
            let lower_value = value.to_lowercase();

            if normalized_label == observation.normalized_surface {
                return true;
            }

            if surface_word_count == 1
                && lower_value.starts_with(&lower_surface)
                && is_loose_note_generic_field_label(&label)
            {
                return true;
            }
        }
    }

    if span
        .text
        .lines()
        .filter(|line| line.trim_start().starts_with("- "))
        .any(|line| {
            first_cleaned_token(line.trim())
                .map(|token| token.to_lowercase() == lower_surface)
                .unwrap_or(false)
        })
        && !observation
            .aggregate_features
            .contains(&MentionFeature::PossessiveObserved)
        && is_loose_note_list_item_singleton(&observation.surface)
    {
        return true;
    }

    false
}

fn follows_loose_note_label_pattern(remaining_text: &str) -> bool {
    matches!(
        remaining_text.split_whitespace().next(),
        Some(
            "with"
                | "to"
                | "of"
                | "for"
                | "friend"
                | "identity"
                | "profile"
                | "summary"
                | "background"
        )
    )
}

fn is_loose_note_generic_field_label(label: &str) -> bool {
    label
        .split_whitespace()
        .any(|word| {
            matches!(
                word.to_ascii_lowercase().as_str(),
                "role"
                    | "history"
                    | "relationship"
                    | "dynamic"
                    | "identity"
                    | "personality"
                    | "purpose"
                    | "tone"
                    | "outcome"
                    | "opening"
                    | "closing"
                    | "summary"
            )
        })
}

fn is_loose_note_list_item_singleton(surface: &str) -> bool {
    matches!(
        surface,
        "Calm"
            | "Critical"
            | "Childhood"
            | "Deep"
            | "Known"
            | "Drives"
            | "Loved"
            | "Idolizes"
            | "Moves"
            | "Often"
            | "Frequently"
            | "Feels"
            | "Shares"
            | "Serves"
            | "Assigned"
            | "Acts"
            | "Accidentally"
            | "Flawless"
            | "Friendly"
            | "Half"
            | "Helps"
            | "Manages"
            | "Mutual"
            | "Overprepares"
            | "Perfect"
            | "Picks"
            | "Quietly"
            | "Speaks"
            | "Sometimes"
            | "Surprisingly"
            | "Energetic"
            | "Genuinely"
    )
}

fn first_cleaned_token(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(clean_entity_token)
        .find(|token| !token.text.is_empty())
        .map(|token| token.text)
}

fn build_occurrence_snippet(text: &str, surface: &str) -> String {
    let normalized_text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let lowercase_text = normalized_text.to_lowercase();
    let lowercase_surface = surface.to_lowercase();

    let Some(match_start) = lowercase_text.find(&lowercase_surface) else {
        return truncate_to_char_limit(&normalized_text, SUMMARY_TEXT_LIMIT);
    };

    let match_end = match_start + lowercase_surface.len();
    let match_start_char = lowercase_text[..match_start].chars().count();
    let match_end_char = lowercase_text[..match_end].chars().count();
    let start_char = match_start_char.saturating_sub(80);
    let end_char = (match_end_char + 120).min(normalized_text.chars().count());

    normalized_text
        .chars()
        .skip(start_char)
        .take(end_char.saturating_sub(start_char))
        .collect::<String>()
        .trim()
        .to_string()
}

fn cooccurring_mentions_in_span(span: &ParsedSpan, surface: &str) -> Vec<String> {
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

        let words = tokens[start_index..index]
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>();
        let mention = words.join(" ");

        if mention.is_empty() || mention == surface || mentions.contains(&mention) {
            continue;
        }

        mentions.push(mention);
    }

    mentions.truncate(4);
    mentions
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
        // Decorative glyphs in labels/values are usually chatty note markup,
        // not stable structured evidence we want to reuse downstream.
        || contains_emoji_characters(&label)
        || contains_emoji_characters(&value)
    {
        return None;
    }

    Some((label, value))
}

fn is_alias_like_label(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "alias" | "aliases" | "nickname" | "callsign"
    )
}

fn is_story_planning_participant_label(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "participant"
            | "participants"
            | "focus"
            | "target"
            | "character"
            | "characters"
            | "crew"
            | "speaker"
            | "speakers"
    )
}

fn split_story_planning_mentions(value: &str) -> Vec<String> {
    value.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .filter(|part| part.chars().any(|character| character.is_alphabetic()))
        .filter(|part| part.split_whitespace().count() <= 3)
        .map(|part| part.to_string())
        .collect()
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
        // Keep terminology/definition evidence text-clean for retrieval and
        // later semantic consolidation.
        || contains_emoji_characters(&term)
        || contains_emoji_characters(&definition)
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

fn merge_occurrences(existing: &mut Vec<MentionOccurrence>, incoming: &[MentionOccurrence]) {
    for occurrence in incoming {
        if existing
            .iter()
            .any(|existing_occurrence| existing_occurrence == occurrence)
        {
            continue;
        }

        if existing.len() >= 3 {
            break;
        }

        existing.push(occurrence.clone());
    }
}

fn merge_features(existing: &mut Vec<MentionFeature>, incoming: &[MentionFeature]) {
    for feature in incoming {
        if !existing.contains(feature) {
            existing.push(feature.clone());
        }
    }
}
