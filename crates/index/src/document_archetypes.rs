use writing_assist_core::{DocumentArchetype, DocumentType, ParsedMarkdownDocument, SpanType};

pub fn classify_document_archetype(
    document_type: DocumentType,
    path: &str,
    text: &str,
    parsed: &ParsedMarkdownDocument,
) -> DocumentArchetype {
    if document_type == DocumentType::Manuscript {
        return DocumentArchetype::Manuscript;
    }

    let features = ArchetypeFeatures::new(path, text, parsed);

    if features.matches_story_planning() {
        return DocumentArchetype::StoryPlanning;
    }

    if document_type == DocumentType::Reference {
        if features.matches_dossier_profile() {
            return DocumentArchetype::DossierProfile;
        }

        if features.matches_taxonomy_reference() {
            return DocumentArchetype::TaxonomyReference;
        }

        // Prefer the more structurally specific reference shape before the broader
        // prose-article fallback, so mixed reference documents do not collapse into
        // "world article" just because they include some explanatory paragraphs.
        if features.matches_expository_world_article() {
            return DocumentArchetype::ExpositoryWorldArticle;
        }
    }

    DocumentArchetype::LooseNote
}

struct ArchetypeFeatures<'a> {
    path_tokens: Vec<&'a str>,
    nonempty_lines: Vec<&'a str>,
    bullet_line_count: usize,
    label_line_count: usize,
    numbered_line_count: usize,
    heading_span_count: usize,
    paragraph_span_count: usize,
    short_line_count: usize,
    long_line_count: usize,
    heading_like_line_count: usize,
    dashed_title_line_count: usize,
    planning_label_count: usize,
    profile_label_count: usize,
    profile_section_heading_count: usize,
    definition_like_line_count: usize,
}

impl<'a> ArchetypeFeatures<'a> {
    fn new(path: &'a str, text: &'a str, parsed: &ParsedMarkdownDocument) -> Self {
        let path_tokens = path
            .split(|character: char| !character.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let nonempty_lines = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        let bullet_line_count = nonempty_lines
            .iter()
            .filter(|line| {
                line.starts_with("- ")
                    || line.starts_with("* ")
                    || line.starts_with("+ ")
                    || line.starts_with("•")
            })
            .count();
        let label_line_count = nonempty_lines
            .iter()
            .filter(|line| is_compact_label_line(line))
            .count();
        let numbered_line_count = nonempty_lines
            .iter()
            .filter(|line| starts_with_number_or_roman_heading(line))
            .count();
        let short_line_count = nonempty_lines
            .iter()
            .filter(|line| line.chars().count() <= 72)
            .count();
        let long_line_count = nonempty_lines
            .iter()
            .filter(|line| line.chars().count() >= 100)
            .count();
        let heading_like_line_count = nonempty_lines
            .iter()
            .filter(|line| is_heading_like_line(line))
            .count();
        let dashed_title_line_count = nonempty_lines
            .iter()
            .filter(|line| is_dashed_title_line(line))
            .count();
        let planning_label_count = nonempty_lines
            .iter()
            .filter(|line| is_planning_label_line(line))
            .count();
        let profile_label_count = nonempty_lines
            .iter()
            .filter(|line| is_profile_label_line(line))
            .count();
        let profile_section_heading_count = nonempty_lines
            .iter()
            .filter(|line| is_profile_section_heading_line(line))
            .count();
        let definition_like_line_count = nonempty_lines
            .iter()
            .filter(|line| is_definition_like_line(line))
            .count();
        let heading_span_count = parsed
            .spans
            .iter()
            .filter(|span| span.span_type == SpanType::Heading)
            .count();
        let paragraph_span_count = parsed
            .spans
            .iter()
            .filter(|span| span.span_type == SpanType::Paragraph)
            .count();

        Self {
            path_tokens,
            nonempty_lines,
            bullet_line_count,
            label_line_count,
            numbered_line_count,
            heading_span_count,
            paragraph_span_count,
            short_line_count,
            long_line_count,
            heading_like_line_count,
            dashed_title_line_count,
            planning_label_count,
            profile_label_count,
            profile_section_heading_count,
            definition_like_line_count,
        }
    }

    fn matches_dossier_profile(&self) -> bool {
        let has_profile_path_hint =
            self.has_any_path_token(&["profile", "profiles", "dossier", "character", "crew"]);

        self.profile_label_count >= 2
            && (self.label_line_count >= 3 || self.profile_section_heading_count >= 2)
            && self.planning_label_count == 0
            && (self.dashed_title_line_count >= 1 || has_profile_path_hint)
            && self.short_line_count * 2 >= self.nonempty_lines.len()
    }

    fn matches_story_planning(&self) -> bool {
        let score = self.has_any_path_token(&["plan", "planning", "outline", "arc", "arcs", "beat", "beats"])
            as usize
            * 2
            + usize::from(self.planning_label_count >= 2) * 3
            + usize::from(self.bullet_line_count >= 2)
            + usize::from(self.numbered_line_count >= 1)
            + usize::from(self.heading_like_line_count >= 2)
            + usize::from(self.short_line_count * 3 >= self.nonempty_lines.len() * 2);

        score >= 6
    }

    fn matches_taxonomy_reference(&self) -> bool {
        let score = self.has_any_path_token(&["glossary", "terminology", "taxonomy", "lexicon"])
            as usize
            * 2
            + usize::from(self.definition_like_line_count >= 2) * 2
            + usize::from(self.bullet_line_count >= 4) * 2
            + usize::from(self.heading_like_line_count >= 2)
            + usize::from(self.long_line_count * 4 <= self.nonempty_lines.len())
            + usize::from(self.short_line_count * 3 >= self.nonempty_lines.len() * 2)
            + usize::from(self.profile_label_count == 0)
            + usize::from(self.profile_section_heading_count == 0)
            + usize::from(self.planning_label_count == 0)
            + usize::from(self.paragraph_span_count <= 3);

        score >= 6
    }

    fn matches_expository_world_article(&self) -> bool {
        let score = self.has_any_path_token(&["history", "politics", "society", "culture", "world"])
            as usize
            * 2
            + usize::from(self.heading_span_count >= 2 || self.heading_like_line_count >= 2) * 2
            + usize::from(self.paragraph_span_count >= 4) * 2
            + usize::from(self.long_line_count >= 2) * 2
            + usize::from(self.bullet_line_count <= 2) * 2
            + usize::from(self.long_line_count > self.bullet_line_count)
            + usize::from(self.paragraph_span_count > self.bullet_line_count) * 2;

        score >= 6
    }

    fn has_any_path_token(&self, expected: &[&str]) -> bool {
        expected.iter().any(|candidate| {
            self.path_tokens
                .iter()
                .any(|token| token.eq_ignore_ascii_case(candidate))
        })
    }

}

fn starts_with_number_or_roman_heading(line: &str) -> bool {
    let trimmed = line.trim_start();

    trimmed
        .chars()
        .next()
        .map(|character| character.is_ascii_digit())
        .unwrap_or(false)
        || trimmed.starts_with("I.")
        || trimmed.starts_with("II.")
        || trimmed.starts_with("III.")
        || trimmed.starts_with("IV.")
        || trimmed.starts_with("V.")
        || trimmed.starts_with("VI.")
        || trimmed.starts_with("VII.")
        || trimmed.starts_with("VIII.")
        || trimmed.starts_with("IX.")
        || trimmed.starts_with("X.")
}

fn is_compact_label_line(line: &str) -> bool {
    let Some((left, right)) = line.split_once(':') else {
        return false;
    };

    let left_word_count = left.split_whitespace().count();
    let right_has_text = right.chars().any(|character| character.is_alphanumeric());

    (1..=4).contains(&left_word_count)
        && right_has_text
        && !left.trim_start().starts_with('#')
        && line.chars().count() <= 120
}

fn is_planning_label_line(line: &&str) -> bool {
    let Some((left, _)) = line.split_once(':') else {
        return false;
    };

    matches!(
        left.trim().to_ascii_lowercase().as_str(),
        "tone"
            | "goal"
            | "purpose"
            | "result"
            | "outcome"
            | "stakes"
            | "structure"
            | "scope"
            | "summary"
            | "sequence"
    )
}

fn is_profile_label_line(line: &&str) -> bool {
    let Some((left, _)) = line.split_once(':') else {
        return false;
    };

    let normalized = left.trim().to_ascii_lowercase();

    normalized.contains("role")
        || normalized.contains("background")
        || normalized.contains("history")
        || normalized.contains("trait")
        || normalized.contains("relationship")
        || normalized.contains("special")
        || normalized.contains("identity")
        || normalized.contains("profession")
        || normalized.contains("rank")
        || normalized.contains("dynamic")
        || normalized.contains("personality")
        || normalized.contains("mission")
        || normalized.contains("summary")
}

fn is_profile_section_heading_line(line: &&str) -> bool {
    let normalized = line
        .trim()
        .trim_matches(|character: char| character == '#' || character == '-' || character == ':')
        .trim()
        .to_ascii_lowercase();

    normalized == "relationships"
        || normalized == "relationship"
        || normalized == "background"
        || normalized == "history"
        || normalized == "personality"
        || normalized == "professional identity"
        || normalized == "professional summary"
        || normalized == "identity"
        || normalized == "mission"
        || normalized == "dynamics"
}

fn is_definition_like_line(line: &&str) -> bool {
    if is_compact_label_line(line) {
        return true;
    }

    let trimmed = line.trim();
    let word_count = trimmed.split_whitespace().count();

    word_count <= 12
        && trimmed.chars().count() <= 120
        && (trimmed.contains(" = ") || trimmed.contains(" — ") || trimmed.contains(" - "))
}

fn is_heading_like_line(line: &&str) -> bool {
    let trimmed = line.trim();

    if trimmed.starts_with('#') || starts_with_number_or_roman_heading(trimmed) {
        return true;
    }

    let word_count = trimmed.split_whitespace().count();
    let uppercase_letters = trimmed.chars().filter(|character| character.is_uppercase()).count();
    let alphabetic_letters = trimmed.chars().filter(|character| character.is_alphabetic()).count();

    alphabetic_letters > 0
        && word_count <= 12
        && trimmed.chars().count() <= 90
        && uppercase_letters * 2 >= alphabetic_letters
}

fn is_dashed_title_line(line: &&str) -> bool {
    let trimmed = line.trim();
    (trimmed.contains(" — ") || trimmed.contains(" - "))
        && trimmed.split_whitespace().count() <= 12
        && trimmed.chars().count() <= 100
}
