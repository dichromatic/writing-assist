use writing_assist_core::{
    ParsedMarkdownDocument, ParsedScene, ParsedSection, ParsedSpan, SpanType,
};

struct SourceLine<'a> {
    text: &'a str,
    start_byte: usize,
    start_char: usize,
}

pub fn supported_span_types() -> [SpanType; 5] {
    [
        SpanType::Heading,
        SpanType::Paragraph,
        SpanType::Section,
        SpanType::Window,
        SpanType::Scene,
    ]
}

fn is_heading_line(line: &str) -> bool {
    let trimmed = line.trim();

    if trimmed.is_empty() {
        return false;
    }

    let heading_markers = trimmed.chars().take_while(|character| *character == '#').count();

    heading_markers > 0 && trimmed.chars().nth(heading_markers) == Some(' ')
}

fn is_scene_break_line(line: &str) -> bool {
    let trimmed = line.trim();

    if trimmed.is_empty() {
        return false;
    }

    let marker: String = trimmed.chars().filter(|character| !character.is_whitespace()).collect();

    if marker.len() < 3 {
        return false;
    }

    let mut characters = marker.chars();
    let Some(first_character) = characters.next() else {
        return false;
    };

    matches!(first_character, '-' | '*' | '_')
        && characters.all(|character| character == first_character)
}

fn ends_with_terminal_punctuation(line: &str) -> bool {
    let trimmed = line.trim_end();

    matches!(
        trimmed.chars().last(),
        Some('.' | '!' | '?' | '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '…')
    )
}

fn starts_like_new_paragraph(line: &str) -> bool {
    let trimmed = line.trim_start();

    let Some(first_character) = trimmed.chars().next() else {
        return false;
    };

    let looks_like_paragraph_body =
        trimmed.len() >= 20 || trimmed.split_whitespace().take(4).count() >= 4;

    looks_like_paragraph_body
        && (first_character.is_uppercase()
            || first_character.is_numeric()
            || matches!(first_character, '"' | '\'' | '(' | '[' | '{' | '“' | '‘' | '—'))
}

fn collect_source_lines(markdown: &str) -> Vec<SourceLine<'_>> {
    let mut lines = Vec::new();
    let mut start_byte = 0;
    let mut start_char = 0;

    for raw_line in markdown.split_inclusive('\n') {
        let text = raw_line.strip_suffix('\n').unwrap_or(raw_line);

        lines.push(SourceLine {
            text,
            start_byte,
            start_char,
        });

        start_byte += raw_line.len();
        start_char += raw_line.chars().count();
    }

    if !markdown.ends_with('\n') && !markdown.is_empty() {
        return lines;
    }

    if markdown.is_empty() {
        return Vec::new();
    }

    lines
}

fn line_end_byte(line: &SourceLine<'_>) -> usize {
    line.start_byte + line.text.len()
}

fn line_end_char(line: &SourceLine<'_>) -> usize {
    line.start_char + line.text.chars().count()
}

fn should_split_paragraph_without_blank_lines(
    paragraph_lines: &[String],
    next_line: &str,
    section_has_explicit_blank_lines: bool,
) -> bool {
    if section_has_explicit_blank_lines || paragraph_lines.is_empty() {
        return false;
    }

    let Some(previous_line) = paragraph_lines.last() else {
        return false;
    };

    // This heuristic is intentionally narrow: only split implicit paragraphs when a section is running
    // without blank lines and the previous line looks complete while the next line looks like a fresh start.
    ends_with_terminal_punctuation(previous_line) && starts_like_new_paragraph(next_line)
}

fn push_paragraph_span(
    spans: &mut Vec<ParsedSpan>,
    source_lines: &[SourceLine<'_>],
    paragraph_lines: &mut Vec<String>,
    paragraph_start_line: &mut Option<usize>,
    current_line_index: usize,
) -> Option<usize> {
    let Some(start_line) = paragraph_start_line.take() else {
        return None;
    };

    if paragraph_lines.is_empty() {
        return None;
    }

    let text = paragraph_lines.join("\n");
    paragraph_lines.clear();
    let end_line = current_line_index.saturating_sub(1);
    let source_start = &source_lines[start_line];
    let source_end = &source_lines[end_line];

    spans.push(ParsedSpan {
        ordinal: spans.len(),
        span_type: SpanType::Paragraph,
        text,
        start_line,
        end_line,
        start_byte: source_start.start_byte,
        end_byte: line_end_byte(source_end),
        start_char: source_start.start_char,
        end_char: line_end_char(source_end),
    });

    spans.last().map(|span| span.ordinal)
}

fn push_section(
    sections: &mut Vec<ParsedSection>,
    source_lines: &[SourceLine<'_>],
    section_lines: &mut Vec<String>,
    section_start_line: &mut Option<usize>,
    current_line_index: usize,
) {
    let Some(start_line) = section_start_line.take() else {
        return;
    };

    let trailing_blank_lines = section_lines
        .iter()
        .rev()
        .take_while(|line| line.is_empty())
        .count();

    while matches!(section_lines.last(), Some(last_line) if last_line.is_empty()) {
        section_lines.pop();
    }

    if section_lines.is_empty() {
        return;
    }

    let end_line = current_line_index.saturating_sub(1 + trailing_blank_lines);
    let source_start = &source_lines[start_line];
    let source_end = &source_lines[end_line];

    sections.push(ParsedSection {
        ordinal: sections.len(),
        text: section_lines.join("\n"),
        start_line,
        end_line,
        start_byte: source_start.start_byte,
        end_byte: line_end_byte(source_end),
        start_char: source_start.start_char,
        end_char: line_end_char(source_end),
    });

    section_lines.clear();
}

fn push_scene(
    scenes: &mut Vec<ParsedScene>,
    source_lines: &[SourceLine<'_>],
    scene_lines: &mut Vec<String>,
    scene_start_line: &mut Option<usize>,
    current_line_index: usize,
    scene_separator: &mut Option<String>,
    scene_start_span_ordinal: &mut Option<usize>,
    scene_end_span_ordinal: &mut Option<usize>,
) {
    let Some(start_line) = scene_start_line.take() else {
        return;
    };

    let trailing_blank_lines = scene_lines
        .iter()
        .rev()
        .take_while(|line| line.is_empty())
        .count();

    while matches!(scene_lines.last(), Some(last_line) if last_line.is_empty()) {
        scene_lines.pop();
    }

    let Some(start_span_ordinal) = scene_start_span_ordinal.take() else {
        scene_lines.clear();
        return;
    };
    let Some(end_span_ordinal) = scene_end_span_ordinal.take() else {
        scene_lines.clear();
        return;
    };

    if scene_lines.is_empty() {
        return;
    }

    let end_line = current_line_index.saturating_sub(1 + trailing_blank_lines);
    let source_start = &source_lines[start_line];
    let source_end = &source_lines[end_line];

    scenes.push(ParsedScene {
        ordinal: scenes.len(),
        text: scene_lines.join("\n"),
        separator: scene_separator.take(),
        start_line,
        end_line,
        start_byte: source_start.start_byte,
        end_byte: line_end_byte(source_end),
        start_char: source_start.start_char,
        end_char: line_end_char(source_end),
        start_span_ordinal,
        end_span_ordinal,
    });

    scene_lines.clear();
}

pub fn parse_markdown_document(markdown: &str) -> ParsedMarkdownDocument {
    let source_lines = collect_source_lines(markdown);

    if source_lines.is_empty() {
        return ParsedMarkdownDocument {
            spans: Vec::new(),
            sections: Vec::new(),
            scenes: Vec::new(),
        };
    }

    let mut spans = Vec::new();
    let mut sections = Vec::new();
    let mut scenes = Vec::new();
    let mut paragraph_lines = Vec::new();
    let mut paragraph_start_line = None;
    let mut section_lines = Vec::new();
    let mut section_start_line = None;
    let mut section_has_explicit_blank_lines = false;
    let mut scene_lines = Vec::new();
    let mut scene_start_line = None;
    let mut scene_separator = None;
    let mut scene_start_span_ordinal = None;
    let mut scene_end_span_ordinal = None;

    for (line_index, source_line) in source_lines.iter().enumerate() {
        let line = source_line.text;

        if is_heading_line(line) {
            if let Some(paragraph_span_ordinal) = push_paragraph_span(
                &mut spans,
                &source_lines,
                &mut paragraph_lines,
                &mut paragraph_start_line,
                line_index,
            ) {
                if scene_start_span_ordinal.is_none() {
                    scene_start_span_ordinal = Some(paragraph_span_ordinal);
                }
            }
            push_section(
                &mut sections,
                &source_lines,
                &mut section_lines,
                &mut section_start_line,
                line_index,
            );

            spans.push(ParsedSpan {
                ordinal: spans.len(),
                span_type: SpanType::Heading,
                text: line.to_string(),
                start_line: line_index,
                end_line: line_index,
                start_byte: source_line.start_byte,
                end_byte: line_end_byte(source_line),
                start_char: source_line.start_char,
                end_char: line_end_char(source_line),
            });

            if scene_start_span_ordinal.is_none() {
                scene_start_span_ordinal = spans.last().map(|span| span.ordinal);
            }
            scene_end_span_ordinal = spans.last().map(|span| span.ordinal);

            section_start_line = Some(line_index);
            section_has_explicit_blank_lines = false;
            section_lines.push(line.to_string());

            if scene_start_line.is_none() {
                scene_start_line = Some(line_index);
            }
            scene_lines.push(line.to_string());
            continue;
        }

        if is_scene_break_line(line) {
            if let Some(paragraph_span_ordinal) = push_paragraph_span(
                &mut spans,
                &source_lines,
                &mut paragraph_lines,
                &mut paragraph_start_line,
                line_index,
            ) {
                if scene_start_span_ordinal.is_none() {
                    scene_start_span_ordinal = Some(paragraph_span_ordinal);
                }
                scene_end_span_ordinal = Some(paragraph_span_ordinal);
            }
            push_section(
                &mut sections,
                &source_lines,
                &mut section_lines,
                &mut section_start_line,
                line_index,
            );
            push_scene(
                &mut scenes,
                &source_lines,
                &mut scene_lines,
                &mut scene_start_line,
                line_index,
                &mut scene_separator,
                &mut scene_start_span_ordinal,
                &mut scene_end_span_ordinal,
            );

            spans.push(ParsedSpan {
                ordinal: spans.len(),
                span_type: SpanType::Scene,
                text: line.trim().to_string(),
                start_line: line_index,
                end_line: line_index,
                start_byte: source_line.start_byte,
                end_byte: line_end_byte(source_line),
                start_char: source_line.start_char,
                end_char: line_end_char(source_line),
            });

            // Treat thematic separators as scene boundaries without folding the separator line into either section body.
            section_has_explicit_blank_lines = false;
            scene_separator = Some(line.trim().to_string());
            continue;
        }

        if line.trim().is_empty() {
            if let Some(paragraph_span_ordinal) = push_paragraph_span(
                &mut spans,
                &source_lines,
                &mut paragraph_lines,
                &mut paragraph_start_line,
                line_index,
            ) {
                if scene_start_span_ordinal.is_none() {
                    scene_start_span_ordinal = Some(paragraph_span_ordinal);
                }
                scene_end_span_ordinal = Some(paragraph_span_ordinal);
            }

            if section_start_line.is_some() {
                section_lines.push(String::new());
                section_has_explicit_blank_lines = true;
            }

            if scene_start_line.is_some() {
                scene_lines.push(String::new());
            }

            continue;
        }

        if section_start_line.is_none() {
            section_start_line = Some(line_index);
        }

        if paragraph_start_line.is_none() {
            paragraph_start_line = Some(line_index);
        }

        if should_split_paragraph_without_blank_lines(
            &paragraph_lines,
            line,
            section_has_explicit_blank_lines,
        ) {
            if let Some(paragraph_span_ordinal) = push_paragraph_span(
                &mut spans,
                &source_lines,
                &mut paragraph_lines,
                &mut paragraph_start_line,
                line_index,
            ) {
                if scene_start_span_ordinal.is_none() {
                    scene_start_span_ordinal = Some(paragraph_span_ordinal);
                }
                scene_end_span_ordinal = Some(paragraph_span_ordinal);
            }
            paragraph_start_line = Some(line_index);
        }

        if scene_start_line.is_none() {
            scene_start_line = Some(line_index);
        }

        paragraph_lines.push(line.to_string());
        section_lines.push(line.to_string());
        scene_lines.push(line.to_string());
    }

    // Flush trailing paragraph/section content so the last block survives when the file lacks a final blank line.
    if let Some(paragraph_span_ordinal) = push_paragraph_span(
        &mut spans,
        &source_lines,
        &mut paragraph_lines,
        &mut paragraph_start_line,
        source_lines.len(),
    ) {
        if scene_start_span_ordinal.is_none() {
            scene_start_span_ordinal = Some(paragraph_span_ordinal);
        }
        scene_end_span_ordinal = Some(paragraph_span_ordinal);
    }
    push_section(
        &mut sections,
        &source_lines,
        &mut section_lines,
        &mut section_start_line,
        source_lines.len(),
    );
    push_scene(
        &mut scenes,
        &source_lines,
        &mut scene_lines,
        &mut scene_start_line,
        source_lines.len(),
        &mut scene_separator,
        &mut scene_start_span_ordinal,
        &mut scene_end_span_ordinal,
    );

    ParsedMarkdownDocument {
        spans,
        sections,
        scenes,
    }
}
