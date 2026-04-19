use writing_assist_core::{
    ParsedMarkdownDocument, ParsedSpan, PreprocessedDocument, PreprocessedQuoteSpan,
    PreprocessedSentence, PreprocessedSpan, PreprocessedToken, SectionBoundaryKind,
    SentenceType, SpanType, StructuralMarker, StructuralMarkerKind,
};

const TOKENIZER_VERSION: &str = "deterministic_v1";
const TITLE_ABBREVIATIONS: &[&str] = &["Dr", "Mr", "Mrs", "Ms", "Prof"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanKind {
    Heading,
    Paragraph,
    ListItem,
}

/// Build a deterministic structural preprocessing view over parsed Markdown.
///
/// This keeps original character offsets intact while producing normalized
/// tokens, sentence boundaries, quote spans, and structural markers that later
/// harvesting and retrieval code can reuse.
pub fn preprocess_parsed_document(parsed: &ParsedMarkdownDocument) -> PreprocessedDocument {
    let mut spans = Vec::new();
    let mut structural_markers = Vec::new();

    for span in &parsed.spans {
        if !matches!(span.span_type, SpanType::Heading | SpanType::Paragraph) {
            continue;
        }

        let structural_kind = classify_span_kind(span);
        spans.push(preprocess_span(span, structural_kind));
        structural_markers.extend(structural_markers_for_span(span, structural_kind));
    }

    structural_markers.extend(parsed.sections.iter().filter_map(|section| {
        if section.boundary_kind != SectionBoundaryKind::SceneBreak {
            return None;
        }

        Some(StructuralMarker {
            kind: StructuralMarkerKind::SceneBreak,
            span_ordinal: None,
            section_ordinal: Some(section.ordinal),
            text: section.boundary_text.clone(),
            start_char: section.start_char,
            end_char: section.start_char,
        })
    }));

    PreprocessedDocument {
        tokenizer_version: TOKENIZER_VERSION.to_string(),
        spans,
        structural_markers,
    }
}

fn preprocess_span(span: &ParsedSpan, structural_kind: SpanKind) -> PreprocessedSpan {
    let tokens = tokenize_span(span);
    let sentences = segment_sentences(span, structural_kind);
    let quote_spans = extract_quote_spans(span);

    PreprocessedSpan {
        span_ordinal: span.ordinal,
        structural_kind: match structural_kind {
            SpanKind::Heading => StructuralMarkerKind::Heading,
            SpanKind::Paragraph => StructuralMarkerKind::Paragraph,
            SpanKind::ListItem => StructuralMarkerKind::ListItem,
        },
        normalized_text: normalize_text_for_matching(&span.text),
        tokens,
        sentences,
        quote_spans,
    }
}

fn classify_span_kind(span: &ParsedSpan) -> SpanKind {
    if span.span_type == SpanType::Heading {
        return SpanKind::Heading;
    }

    if span
        .text
        .lines()
        .any(|line| is_list_item_line(line.trim_start()))
    {
        SpanKind::ListItem
    } else {
        SpanKind::Paragraph
    }
}

fn structural_markers_for_span(span: &ParsedSpan, structural_kind: SpanKind) -> Vec<StructuralMarker> {
    match structural_kind {
        SpanKind::Heading => vec![StructuralMarker {
            kind: StructuralMarkerKind::Heading,
            span_ordinal: Some(span.ordinal),
            section_ordinal: None,
            text: Some(span.text.clone()),
            start_char: span.start_char,
            end_char: span.end_char,
        }],
        SpanKind::Paragraph => Vec::new(),
        SpanKind::ListItem => {
            let mut markers = Vec::new();
            let mut local_start_char = 0;

            for line in span.text.lines() {
                let line_char_len = line.chars().count();
                if is_list_item_line(line.trim_start()) {
                    markers.push(StructuralMarker {
                        kind: StructuralMarkerKind::ListItem,
                        span_ordinal: Some(span.ordinal),
                        section_ordinal: None,
                        text: Some(line.trim().to_string()),
                        start_char: span.start_char + local_start_char,
                        end_char: span.start_char + local_start_char + line_char_len,
                    });
                }

                local_start_char += line_char_len + 1;
            }

            markers
        }
    }
}

fn tokenize_span(span: &ParsedSpan) -> Vec<PreprocessedToken> {
    let characters = span.text.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < characters.len() {
        let character = characters[index];

        if character.is_whitespace() {
            index += 1;
            continue;
        }

        if is_word_character(&characters, index) {
            let start = index;
            let mut surface = String::new();

            while index < characters.len() && is_word_character(&characters, index) {
                surface.push(characters[index]);
                index += 1;
            }

            tokens.push(PreprocessedToken {
                normalized: normalize_text_for_matching(&surface),
                surface,
                start_char: span.start_char + start,
                end_char: span.start_char + index,
            });
            continue;
        }

        let surface = character.to_string();
        tokens.push(PreprocessedToken {
            normalized: normalize_inline_text(&surface),
            surface,
            start_char: span.start_char + index,
            end_char: span.start_char + index + 1,
        });
        index += 1;
    }

    tokens
}

fn segment_sentences(span: &ParsedSpan, structural_kind: SpanKind) -> Vec<PreprocessedSentence> {
    match structural_kind {
        SpanKind::Heading => vec![build_sentence(
            span,
            0,
            0,
            span.text.chars().count(),
            SentenceType::Heading,
        )],
        SpanKind::ListItem => segment_list_item_sentences(span),
        SpanKind::Paragraph => segment_paragraph_sentences(span),
    }
}

fn segment_list_item_sentences(span: &ParsedSpan) -> Vec<PreprocessedSentence> {
    let mut sentences = Vec::new();
    let mut local_start_char = 0;

    for line in span.text.lines() {
        let line_char_len = line.chars().count();
        let trimmed = line.trim();

        if !trimmed.is_empty() {
            let sentence_type = if trimmed.starts_with('>') {
                SentenceType::BlockQuote
            } else {
                SentenceType::ListItem
            };

            sentences.push(build_sentence(
                span,
                sentences.len(),
                local_start_char,
                local_start_char + line_char_len,
                sentence_type,
            ));
        }

        local_start_char += line_char_len + 1;
    }

    sentences
}

fn segment_paragraph_sentences(span: &ParsedSpan) -> Vec<PreprocessedSentence> {
    let characters = span.text.chars().collect::<Vec<_>>();
    let mut sentences = Vec::new();
    let mut sentence_start = 0;
    let mut index = 0;

    while index < characters.len() {
        if is_sentence_terminal(&characters, index) {
            let mut sentence_end = index + 1;

            while sentence_end < characters.len() && is_sentence_closer(characters[sentence_end]) {
                sentence_end += 1;
            }

            sentences.push(build_sentence(
                span,
                sentences.len(),
                sentence_start,
                sentence_end,
                classify_sentence_type_from_text(&slice_chars(&characters, sentence_start, sentence_end)),
            ));

            sentence_start = sentence_end;
            while sentence_start < characters.len() && characters[sentence_start].is_whitespace() {
                sentence_start += 1;
            }
            index = sentence_start;
            continue;
        }

        index += 1;
    }

    if sentence_start < characters.len() {
        sentences.push(build_sentence(
            span,
            sentences.len(),
            sentence_start,
            characters.len(),
            classify_sentence_type_from_text(&slice_chars(&characters, sentence_start, characters.len())),
        ));
    }

    sentences
}

fn build_sentence(
    span: &ParsedSpan,
    ordinal: usize,
    local_start_char: usize,
    local_end_char: usize,
    sentence_type: SentenceType,
) -> PreprocessedSentence {
    let characters = span.text.chars().collect::<Vec<_>>();
    let text = slice_chars(&characters, local_start_char, local_end_char)
        .trim()
        .to_string();

    PreprocessedSentence {
        ordinal,
        normalized_text: normalize_text_for_matching(&text),
        text,
        start_char: span.start_char + local_start_char,
        end_char: span.start_char + local_end_char,
        sentence_type,
    }
}

fn extract_quote_spans(span: &ParsedSpan) -> Vec<PreprocessedQuoteSpan> {
    let characters = span.text.chars().collect::<Vec<_>>();
    let mut quotes = Vec::new();
    let mut current_quote_start = None;

    for (index, _character) in characters.iter().enumerate() {
        if !is_quote_character(&characters, index) {
            continue;
        }

        if let Some(start) = current_quote_start.take() {
            let end = index + 1;
            let text = slice_chars(&characters, start, end);
            quotes.push(PreprocessedQuoteSpan {
                ordinal: quotes.len(),
                normalized_text: normalize_inline_text(&text),
                text,
                start_char: span.start_char + start,
                end_char: span.start_char + end,
            });
        } else {
            current_quote_start = Some(index);
        }
    }

    quotes
}

fn is_word_character(characters: &[char], index: usize) -> bool {
    let character = characters[index];

    if character.is_alphanumeric() {
        return true;
    }

    matches!(character, '\'' | '’' | '-' | '‑')
        && index > 0
        && index + 1 < characters.len()
        && characters[index - 1].is_alphanumeric()
        && characters[index + 1].is_alphanumeric()
}

fn is_quote_character(characters: &[char], index: usize) -> bool {
    let character = characters[index];

    matches!(character, '"' | '“' | '”')
        || (matches!(character, '\'' | '‘' | '’')
            && !(index > 0
                && index + 1 < characters.len()
                && characters[index - 1].is_alphanumeric()
                && characters[index + 1].is_alphanumeric()))
}

fn is_sentence_terminal(characters: &[char], index: usize) -> bool {
    match characters[index] {
        // Honorific abbreviations like `Mrs.` and `Dr.` are lexical tokens, not
        // sentence boundaries. The preprocessing layer needs to preserve that
        // distinction so later evidence harvesting can keep titled mentions
        // intact instead of rebuilding around broken sentence slices.
        '.' => !is_title_abbreviation_period(characters, index),
        '!' | '?' | '…' => true,
        _ => false,
    }
}

fn is_sentence_closer(character: char) -> bool {
    matches!(character, '"' | '“' | '”' | '\'' | '‘' | '’' | ')' | ']' | '}')
}

fn is_list_item_line(line: &str) -> bool {
    line.starts_with("- ") || line.starts_with("* ") || line.starts_with("+ ") || starts_with_numbered_list_item(line)
}

fn starts_with_numbered_list_item(text: &str) -> bool {
    let mut saw_digit = false;

    for character in text.chars() {
        if character.is_ascii_digit() {
            saw_digit = true;
            continue;
        }

        return saw_digit && character == '.';
    }

    false
}

fn classify_sentence_type_from_text(text: &str) -> SentenceType {
    let trimmed = text.trim_start();

    if trimmed.starts_with('>') {
        return SentenceType::BlockQuote;
    }

    if trimmed.starts_with('"')
        || trimmed.starts_with('“')
        || trimmed.starts_with('‘')
        || trimmed.starts_with('\'')
        || trimmed.starts_with('—')
        || trimmed.starts_with('-')
    {
        return SentenceType::Dialogue;
    }

    SentenceType::Narrative
}

fn is_title_abbreviation_period(characters: &[char], index: usize) -> bool {
    let Some(word) = previous_word_before_index(characters, index) else {
        return false;
    };

    TITLE_ABBREVIATIONS.contains(&word.as_str())
}

fn previous_word_before_index(characters: &[char], index: usize) -> Option<String> {
    if index == 0 {
        return None;
    }

    let mut end = index;
    while end > 0 && characters[end - 1].is_whitespace() {
        end -= 1;
    }

    let mut start = end;
    while start > 0 && characters[start - 1].is_alphabetic() {
        start -= 1;
    }

    if start == end {
        return None;
    }

    Some(slice_chars(characters, start, end))
}

fn normalize_text_for_matching(text: &str) -> String {
    let mut normalized = String::new();
    let mut previous_was_space = false;

    for character in text.chars() {
        let replacement = normalize_character(character);

        for replacement_character in replacement.chars() {
            if replacement_character.is_whitespace() {
                if !previous_was_space {
                    normalized.push(' ');
                }
                previous_was_space = true;
            } else {
                normalized.push(replacement_character);
                previous_was_space = false;
            }
        }
    }

    normalized.trim().to_string()
}

fn normalize_inline_text(text: &str) -> String {
    text.chars()
        .flat_map(|character| normalize_character(character).chars().collect::<Vec<_>>())
        .collect()
}

fn normalize_character(character: char) -> String {
    match character {
        '“' | '”' => "\"".to_string(),
        '‘' | '’' => "'".to_string(),
        '—' | '–' | '‑' => "-".to_string(),
        '…' => "...".to_string(),
        _ => character.to_string(),
    }
}

fn slice_chars(characters: &[char], start: usize, end: usize) -> String {
    characters[start..end].iter().collect()
}
