use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use uuid::Uuid;
use writing_assist_core::{
    DefinitionCandidate, DocumentArchetype, EvidenceContext, MemorySourceReference,
    MentionCandidate, MentionFeature, MentionOccurrence, ParsedMarkdownDocument, ParsedSection,
    ParsedSpan, PreprocessedDocument, PreprocessedSentence, PreprocessedSpan, PreprocessedToken,
    SectionSummarySeed, SentenceType, SpanType, StructuredFieldCandidate, TargetAnchor,
};

use crate::preprocess_parsed_document;

const MAX_MENTION_WORDS: usize = 5;
const SUMMARY_TEXT_LIMIT: usize = 240;
static ENGLISH_STOPWORDS: OnceLock<HashSet<String>> = OnceLock::new();

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
    let preprocessed = preprocess_parsed_document(parsed);
    let mut observations = Vec::<MentionObservation>::new();
    let mut index_by_normalized_surface = HashMap::<String, usize>::new();

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

        for harvested in mention_observations_in_span(
            document_path,
            span,
            preprocessed_span,
            &preprocessed,
            parsed,
            &archetype,
        ) {
            if let Some(existing_index) = index_by_normalized_surface
                .get(&harvested.normalized_surface)
                .copied()
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
    preprocessed_span: &PreprocessedSpan,
    preprocessed: &PreprocessedDocument,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    match archetype {
        DocumentArchetype::Manuscript => {
            capitalized_mentions_in_span(document_path, span, preprocessed_span, parsed, archetype)
        }
        DocumentArchetype::DossierProfile => {
            let mut observations = capitalized_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
                archetype,
            );
            observations.extend(alias_field_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
            ));
            observations
        }
        DocumentArchetype::TaxonomyReference | DocumentArchetype::ExpositoryWorldArticle => {
            let mut observations = capitalized_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
                archetype,
            );
            observations.extend(definition_term_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
            ));
            observations
        }
        DocumentArchetype::StoryPlanning => {
            let mut observations = capitalized_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
                archetype,
            );
            observations.extend(story_planning_field_mentions_in_span(
                document_path,
                span,
                preprocessed_span,
                parsed,
            ));
            observations
        }
        DocumentArchetype::LooseNote => loose_note_mentions_in_span(
            document_path,
            span,
            preprocessed_span,
            preprocessed,
            parsed,
            archetype,
        ),
    }
}

fn loose_note_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    preprocessed_span: &PreprocessedSpan,
    _preprocessed: &PreprocessedDocument,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    capitalized_mentions_in_span(document_path, span, preprocessed_span, parsed, archetype)
        .into_iter()
        .filter(|observation| !should_reject_loose_note_observation(observation, span))
        .collect()
}

fn capitalized_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    preprocessed_span: &PreprocessedSpan,
    parsed: &ParsedMarkdownDocument,
    archetype: &DocumentArchetype,
) -> Vec<MentionObservation> {
    let mut mentions = Vec::new();

    for sentence in &preprocessed_span.sentences {
        let sentence_tokens = sentence_tokens(preprocessed_span, sentence);
        let mut index = 0;

        while index < sentence_tokens.len() {
            let Some(current_word) = cleaned_word_token(sentence_tokens[index]) else {
                index += 1;
                continue;
            };

            if !is_mention_token(&current_word.text) {
                index += 1;
                continue;
            }

            let start_index = index;
            let mut current_index = index + 1;
            let mut words = vec![current_word];

            while current_index < sentence_tokens.len() {
                if sentence_tokens[current_index].normalized == "."
                    && words
                        .last()
                        .map(|word| is_title_prefix(&word.text))
                        .unwrap_or(false)
                    && current_index + 1 < sentence_tokens.len()
                {
                    if let Some(next_word) = cleaned_word_token(sentence_tokens[current_index + 1])
                    {
                        if is_mention_token(&next_word.text) {
                            words.push(next_word);
                            current_index += 2;
                            continue;
                        }
                    }
                }

                let Some(next_word) = cleaned_word_token(sentence_tokens[current_index]) else {
                    break;
                };

                if !is_mention_token(&next_word.text) {
                    break;
                }

                words.push(next_word);
                current_index += 1;

                if words.len() >= MAX_MENTION_WORDS {
                    break;
                }
            }

            while words.len() > 1 && is_leading_drop_token(&words[0].text) {
                words.remove(0);
            }
            while words.len() > 1 && is_trailing_drop_token(&words[words.len() - 1].text) {
                words.pop();
            }

            if words.is_empty() || words.len() > MAX_MENTION_WORDS {
                index = start_index + 1;
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
                index = start_index + 1;
                continue;
            }

            let mut aggregate_features = aggregate_features_for_surface(span, &surface);
            if words.iter().any(|word| word.had_possessive)
                && !aggregate_features.contains(&MentionFeature::PossessiveObserved)
            {
                aggregate_features.push(MentionFeature::PossessiveObserved);
            }

            let observation = build_surface_observation(
                document_path,
                span,
                parsed,
                sentence,
                &sentence_tokens,
                surface,
                aggregate_features,
            );
            if should_reject_structural_observation(span, sentence, &observation, archetype) {
                index = current_index.max(start_index + 1);
                continue;
            }

            mentions.push(observation);
            index = current_index.max(start_index + 1);
        }
    }

    mentions
}

fn alias_field_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    preprocessed_span: &PreprocessedSpan,
    parsed: &ParsedMarkdownDocument,
) -> Vec<MentionObservation> {
    let mut mentions = Vec::new();

    for line in span.text.lines() {
        let Some((label, value)) = parse_structured_field_line(line) else {
            continue;
        };

        if !is_alias_like_label(&label) || !value.chars().any(|character| character.is_alphabetic())
        {
            continue;
        }

        if contains_emoji_characters(&value) {
            continue;
        }

        let (sentence, sentence_tokens) =
            supporting_sentence_for_surface(preprocessed_span, &value);
        mentions.push(build_surface_observation(
            document_path,
            span,
            parsed,
            sentence,
            &sentence_tokens,
            value.clone(),
            aggregate_features_for_surface(span, &value),
        ));
    }

    mentions
}

fn definition_term_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    preprocessed_span: &PreprocessedSpan,
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

        let (sentence, sentence_tokens) = supporting_sentence_for_surface(preprocessed_span, &term);
        mentions.push(build_surface_observation(
            document_path,
            span,
            parsed,
            sentence,
            &sentence_tokens,
            term.clone(),
            aggregate_features_for_surface(span, &term),
        ));
    }

    mentions
}

fn story_planning_field_mentions_in_span(
    document_path: &str,
    span: &ParsedSpan,
    preprocessed_span: &PreprocessedSpan,
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

            let (sentence, sentence_tokens) =
                supporting_sentence_for_surface(preprocessed_span, &surface);
            mentions.push(build_surface_observation(
                document_path,
                span,
                parsed,
                sentence,
                &sentence_tokens,
                surface.clone(),
                aggregate_features_for_surface(span, &surface),
            ));
        }
    }

    mentions
}

fn build_context(span: &ParsedSpan, parsed: &ParsedMarkdownDocument) -> EvidenceContext {
    let section = parsed
        .sections
        .iter()
        .find(|section| span.start_char >= section.start_char && span.end_char <= section.end_char);

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
    sentence: &PreprocessedSentence,
    sentence_tokens: &[&PreprocessedToken],
    surface: &str,
) -> MentionOccurrence {
    let section = parsed
        .sections
        .iter()
        .find(|section| span.start_char >= section.start_char && span.end_char <= section.end_char);

    MentionOccurrence {
        span_anchor: TargetAnchor::span(span.ordinal),
        section_anchor: section.map(|section| TargetAnchor::section(section.ordinal)),
        heading: section.and_then(section_heading),
        snippet: build_occurrence_snippet(&sentence.normalized_text, surface),
        sentence_type: sentence.sentence_type.clone(),
        cooccurring_mentions: cooccurring_mentions_in_sentence(sentence_tokens, surface),
    }
}

fn build_surface_observation(
    document_path: &str,
    span: &ParsedSpan,
    parsed: &ParsedMarkdownDocument,
    sentence: &PreprocessedSentence,
    sentence_tokens: &[&PreprocessedToken],
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
        occurrences: vec![build_occurrence(
            span,
            parsed,
            sentence,
            sentence_tokens,
            &surface,
        )],
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
    }
}

fn is_mention_token(token: &str) -> bool {
    let Some(first_character) = token.chars().next() else {
        return false;
    };

    first_character.is_uppercase() && token.chars().any(|character| character.is_alphabetic())
}

fn is_leading_drop_token(token: &str) -> bool {
    matches!(
        token,
        "The"
            | "A"
            | "An"
            | "Hey"
            | "Oh"
            | "Ah"
            | "Well"
            | "Yes"
            | "No"
            | "Please"
            | "But"
            | "And"
            | "So"
            | "Though"
            | "When"
            | "While"
            | "After"
            | "Before"
            | "As"
            | "How"
            | "Since"
            | "Wait"
    )
}

fn is_trailing_drop_token(token: &str) -> bool {
    is_common_stopword(token) || is_non_stopword_noise_singleton(token)
}

fn is_noise_singleton_for_archetype(token: &str, archetype: &DocumentArchetype) -> bool {
    is_non_stopword_noise_singleton(token)
        || matches!(archetype, DocumentArchetype::Manuscript) && is_common_stopword(token)
}

fn is_common_stopword(token: &str) -> bool {
    english_stopwords().contains(&normalize_noise_token(token))
}

fn english_stopwords() -> &'static HashSet<String> {
    ENGLISH_STOPWORDS.get_or_init(|| {
        stop_words::get(stop_words::LANGUAGE::English)
            .into_iter()
            .map(normalize_noise_token)
            .collect()
    })
}

fn normalize_noise_token(token: impl AsRef<str>) -> String {
    token
        .as_ref()
        .trim()
        .replace(['’', '‘'], "'")
        .to_lowercase()
}

fn is_non_stopword_noise_singleton(token: &str) -> bool {
    // Keep this supplemental set small and language-general. The main singleton
    // filter should come from standard stopwords plus structural support rules,
    // not project-specific words copied from corpus logs.
    matches!(
        normalize_noise_token(token).as_str(),
        "wait"
            | "hey"
            | "yeah"
            | "i'm"
            | "i'll"
            | "it's"
            | "we're"
            | "we've"
            | "that's"
            | "you're"
            | "don't"
            | "i've"
            | "i-i"
    )
}

fn is_stutter_fragment(token: &str) -> bool {
    let mut characters = token.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    let Some(second) = characters.next() else {
        return false;
    };
    let Some(third) = characters.next() else {
        return false;
    };

    first.is_uppercase() && second == '-' && third.is_alphabetic()
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
        .all(|token| is_noise_singleton_for_archetype(token, archetype))
    {
        return false;
    }

    match archetype {
        DocumentArchetype::Manuscript => {
            observation.occurrences.len() > 1
                || observation.aggregate_features.iter().any(|feature| {
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
                && is_loose_note_list_item_singleton(&observation.surface)
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

    if word_count == 1 && is_noise_singleton_for_archetype(surface, archetype) {
        return true;
    }

    if contains_emoji_characters(surface) {
        return true;
    }

    if normalized_surface
        .split_whitespace()
        .all(|token| is_noise_singleton_for_archetype(token, archetype))
    {
        return true;
    }

    match archetype {
        DocumentArchetype::Manuscript => {
            if is_stutter_fragment(surface) {
                return true;
            }

            if word_count == 1 && titled {
                return true;
            }

            if word_count > 1
                && surface
                    .split_whitespace()
                    .next()
                    .map(is_leading_drop_token)
                    .unwrap_or(false)
            {
                return true;
            }

            !(word_count > 1 || titled || !is_noise_singleton_for_archetype(surface, archetype))
        }
        _ => false,
    }
}

fn should_reject_structural_observation(
    span: &ParsedSpan,
    sentence: &PreprocessedSentence,
    observation: &MentionObservation,
    archetype: &DocumentArchetype,
) -> bool {
    let word_count = observation.surface.split_whitespace().count();
    let has_strong_signal = observation.aggregate_features.iter().any(|feature| {
        matches!(
            feature,
            MentionFeature::MultiWord | MentionFeature::Titled | MentionFeature::PossessiveObserved
        )
    });

    if word_count == 1
        && !matches!(
            archetype,
            DocumentArchetype::LooseNote | DocumentArchetype::DossierProfile
        )
        && is_common_stopword(&observation.surface)
    {
        return true;
    }

    match archetype {
        DocumentArchetype::StoryPlanning => {
            if sentence_is_bracketed_scene_marker(&sentence.normalized_text) {
                return true;
            }

            if word_count == 1
                && (sentence_is_colon_terminated_label(&sentence.normalized_text)
                    || span_has_structured_field_label(span, &observation.normalized_surface))
            {
                return true;
            }

            if word_count == 1
                && sentence.sentence_type == SentenceType::Heading
                && !has_strong_signal
            {
                return true;
            }

            if is_shouty_marker_surface(&observation.surface)
                && matches!(
                    sentence.sentence_type,
                    SentenceType::Heading | SentenceType::ListItem
                )
            {
                return true;
            }
        }
        DocumentArchetype::TaxonomyReference | DocumentArchetype::ExpositoryWorldArticle => {
            if word_count == 1 && sentence_is_colon_terminated_label(&sentence.normalized_text) {
                return true;
            }

            if sentence_has_outline_enumeration(&sentence.normalized_text)
                && (word_count == 1 || surface_has_roman_enumeration_prefix(&observation.surface))
            {
                return true;
            }

            if word_count == 1 && is_roman_numeral_token(&observation.surface) {
                return true;
            }
        }
        _ => {}
    }

    false
}

fn should_reject_loose_note_observation(
    observation: &MentionObservation,
    span: &ParsedSpan,
) -> bool {
    let surface_word_count = observation.surface.split_whitespace().count();
    let lower_surface = observation.surface.to_lowercase();

    for line in span.text.lines() {
        let normalized_line = line.trim().split_whitespace().collect::<Vec<_>>().join(" ");
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
    label.split_whitespace().any(|word| {
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

fn span_has_structured_field_label(span: &ParsedSpan, normalized_surface: &str) -> bool {
    span.text.lines().any(|line| {
        parse_structured_field_line(line)
            .map(|(label, _)| normalize_mention_surface(&label) == normalized_surface)
            .unwrap_or(false)
    })
}

fn sentence_is_colon_terminated_label(sentence_text: &str) -> bool {
    let trimmed = sentence_text.trim();
    trimmed.ends_with(':') || trimmed.ends_with("):") || trimmed.ends_with("**:")
}

fn sentence_is_bracketed_scene_marker(sentence_text: &str) -> bool {
    let trimmed = sentence_text.trim();
    trimmed.starts_with('[') && trimmed.ends_with(']')
}

fn is_shouty_marker_surface(surface: &str) -> bool {
    let mut saw_alpha = false;
    for character in surface
        .chars()
        .filter(|character| character.is_alphabetic())
    {
        saw_alpha = true;
        if !character.is_uppercase() {
            return false;
        }
    }

    saw_alpha
}

fn sentence_has_outline_enumeration(sentence_text: &str) -> bool {
    let trimmed = sentence_text.trim();
    let mut parts = trimmed.split_whitespace();
    let Some(first) = parts.next() else {
        return false;
    };
    let second = parts.next();
    let third = parts.next();

    if is_roman_numeral_token(first.trim_end_matches('.')) {
        return true;
    }

    second
        .zip(third)
        .map(|(second, third)| {
            second.chars().all(|character| character.is_ascii_digit()) && third == "-"
        })
        .unwrap_or(false)
}

fn surface_has_roman_enumeration_prefix(surface: &str) -> bool {
    let mut parts = surface.split_whitespace();
    let Some(first) = parts.next() else {
        return false;
    };

    is_roman_numeral_token(first.trim_end_matches('.'))
}

fn is_roman_numeral_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|character| matches!(character, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
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

fn cooccurring_mentions_in_sentence(
    sentence_tokens: &[&PreprocessedToken],
    surface: &str,
) -> Vec<String> {
    let mut mentions = Vec::new();
    let mut index = 0;

    while index < sentence_tokens.len() {
        let Some(current_word) = cleaned_word_token(sentence_tokens[index]) else {
            index += 1;
            continue;
        };

        if !is_mention_token(&current_word.text) {
            index += 1;
            continue;
        }

        let start_index = index;
        index += 1;

        let mut words = vec![current_word];

        while index < sentence_tokens.len() {
            let Some(next_word) = cleaned_word_token(sentence_tokens[index]) else {
                break;
            };

            if !is_mention_token(&next_word.text) {
                break;
            }

            words.push(next_word);
            index += 1;
        }

        let mention = words
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if mention.is_empty()
            || mention == surface
            || mentions.contains(&mention)
            || is_stutter_fragment(&mention)
        {
            index = start_index + 1;
            continue;
        }

        mentions.push(mention);
    }

    mentions.truncate(4);
    mentions
}

fn sentence_tokens<'a>(
    preprocessed_span: &'a PreprocessedSpan,
    sentence: &PreprocessedSentence,
) -> Vec<&'a PreprocessedToken> {
    preprocessed_span
        .tokens
        .iter()
        .filter(|token| {
            token.start_char >= sentence.start_char && token.end_char <= sentence.end_char
        })
        .collect()
}

fn supporting_sentence_for_surface<'a>(
    preprocessed_span: &'a PreprocessedSpan,
    surface: &str,
) -> (&'a PreprocessedSentence, Vec<&'a PreprocessedToken>) {
    let normalized_surface = normalize_text(surface);

    if let Some(sentence) = preprocessed_span
        .sentences
        .iter()
        .find(|sentence| sentence.normalized_text.contains(&normalized_surface))
    {
        let sentence_tokens = sentence_tokens(preprocessed_span, sentence);
        return (sentence, sentence_tokens);
    }

    let sentence = preprocessed_span
        .sentences
        .first()
        .expect("preprocessed spans should retain at least one sentence");
    let sentence_tokens = sentence_tokens(preprocessed_span, sentence);
    (sentence, sentence_tokens)
}

fn cleaned_word_token(token: &PreprocessedToken) -> Option<TokenObservation> {
    if token
        .surface
        .chars()
        .all(|character| character.is_whitespace() || !character.is_alphanumeric())
    {
        return None;
    }

    let (text, had_possessive) = if let Some(stripped) = token.normalized.strip_suffix("'s") {
        (stripped, true)
    } else {
        (token.normalized.as_str(), false)
    };

    if text.is_empty() {
        return None;
    }

    Some(TokenObservation {
        text: text.to_string(),
        had_possessive,
    })
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
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
    value
        .split(',')
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
        .trim_matches(|character: char| matches!(character, ':' | '-' | '—' | '*' | '_' | '`'))
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
