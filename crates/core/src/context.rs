use serde::{Deserialize, Serialize};

use crate::{conversation::ConversationMode, documents::DocumentType};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GuideKind {
    Prose,
    Style,
    Critique,
    Rewrite,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    StorySummary,
    WorldSummary,
    CharacterBible,
    Timeline,
    Terminology,
    Research,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "source_type", content = "source_kind", rename_all = "snake_case")]
pub enum ContextSourceKind {
    Guide(GuideKind),
    Reference(ReferenceKind),
    Note,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextSourceActivationPolicy {
    Pinned,
    Retrieved,
    ExplicitOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextSourceReviewState {
    UserAuthored,
    PendingReview,
    Approved,
    Stale,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextSource {
    // Project-relative document path; directory roles remain broad, while this captures document-level semantics.
    pub path: String,
    pub kind: ContextSourceKind,
    pub activation_policy: ContextSourceActivationPolicy,
    pub review_state: ContextSourceReviewState,
}

pub fn classify_context_source_kind(
    document_type: DocumentType,
    path: &str,
    explicit_kind: Option<ContextSourceKind>,
) -> Option<ContextSourceKind> {
    if let Some(explicit_kind) = explicit_kind {
        return Some(explicit_kind);
    }

    match document_type {
        DocumentType::Manuscript => None,
        DocumentType::Note => Some(ContextSourceKind::Note),
        DocumentType::Reference => infer_context_source_kind_from_path(path),
    }
}

fn infer_context_source_kind_from_path(path: &str) -> Option<ContextSourceKind> {
    let normalized = path
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();

    if tokens.is_empty() {
        return None;
    }

    if has_all_tokens(&tokens, &["prose"]) && has_any_token(&tokens, &["guide", "guideline"]) {
        return Some(ContextSourceKind::Guide(GuideKind::Prose));
    }

    if has_all_tokens(&tokens, &["style"]) && has_any_token(&tokens, &["guide", "sheet"]) {
        return Some(ContextSourceKind::Guide(GuideKind::Style));
    }

    if has_all_tokens(&tokens, &["critique"]) && has_any_token(&tokens, &["guide", "rubric"]) {
        return Some(ContextSourceKind::Guide(GuideKind::Critique));
    }

    if has_all_tokens(&tokens, &["rewrite"]) && has_any_token(&tokens, &["guide", "brief"]) {
        return Some(ContextSourceKind::Guide(GuideKind::Rewrite));
    }

    if has_all_tokens(&tokens, &["story"]) && has_any_token(&tokens, &["summary", "synopsis"]) {
        return Some(ContextSourceKind::Reference(ReferenceKind::StorySummary));
    }

    if has_all_tokens(&tokens, &["world"]) && has_any_token(&tokens, &["summary", "overview"]) {
        return Some(ContextSourceKind::Reference(ReferenceKind::WorldSummary));
    }

    if has_any_token(&tokens, &["character", "characters"])
        && has_any_token(&tokens, &["bible", "sheet", "sheets", "guide", "notes"])
    {
        return Some(ContextSourceKind::Reference(ReferenceKind::CharacterBible));
    }

    if has_any_token(&tokens, &["timeline", "chronology"]) {
        return Some(ContextSourceKind::Reference(ReferenceKind::Timeline));
    }

    if has_any_token(&tokens, &["terminology", "glossary", "lexicon", "terms"]) {
        return Some(ContextSourceKind::Reference(ReferenceKind::Terminology));
    }

    if has_any_token(&tokens, &["research", "sourcebook"]) {
        return Some(ContextSourceKind::Reference(ReferenceKind::Research));
    }

    None
}

fn has_any_token(tokens: &[&str], expected: &[&str]) -> bool {
    expected
        .iter()
        .any(|candidate| tokens.iter().any(|token| token == candidate))
}

fn has_all_tokens(tokens: &[&str], expected: &[&str]) -> bool {
    expected
        .iter()
        .all(|candidate| tokens.iter().any(|token| token == candidate))
}

pub fn context_source_allowed_by_default(
    mode: ConversationMode,
    source_kind: &ContextSourceKind,
) -> bool {
    match mode {
        ConversationMode::Analysis => matches!(
            source_kind,
            ContextSourceKind::Guide(_) | ContextSourceKind::Reference(_)
        ),
        ConversationMode::Editing => matches!(
            source_kind,
            ContextSourceKind::Guide(_)
                | ContextSourceKind::Reference(
                    ReferenceKind::CharacterBible
                        | ReferenceKind::Timeline
                        | ReferenceKind::Terminology
                        | ReferenceKind::StorySummary
                        | ReferenceKind::WorldSummary
                )
        ),
        ConversationMode::Ideation => matches!(
            source_kind,
            ContextSourceKind::Guide(_)
                | ContextSourceKind::Reference(
                    ReferenceKind::StorySummary
                        | ReferenceKind::WorldSummary
                        | ReferenceKind::CharacterBible
                )
        ),
    }
}

pub fn context_source_included_by_default(
    mode: ConversationMode,
    source: &ContextSource,
) -> bool {
    let activation_allows_default_use = matches!(
        source.activation_policy,
        ContextSourceActivationPolicy::Pinned | ContextSourceActivationPolicy::Retrieved
    );
    let review_allows_default_use = matches!(
        source.review_state,
        ContextSourceReviewState::UserAuthored | ContextSourceReviewState::Approved
    );

    // Phase 2 task context selection must gate kind-level defaults by review and activation state.
    activation_allows_default_use
        && review_allows_default_use
        && context_source_allowed_by_default(mode, &source.kind)
}

#[cfg(test)]
mod tests {
    use super::{
        classify_context_source_kind, context_source_allowed_by_default,
        context_source_included_by_default, ContextSource, ContextSourceActivationPolicy,
        ContextSourceKind, ContextSourceReviewState, GuideKind, ReferenceKind,
    };
    use crate::{conversation::ConversationMode, documents::DocumentType};

    #[test]
    fn serializes_context_source_kinds_with_explicit_type_and_kind() {
        let guide = serde_json::to_string(&ContextSourceKind::Guide(GuideKind::Prose))
            .expect("serialize guide source kind");
        let reference = serde_json::to_string(&ContextSourceKind::Reference(
            ReferenceKind::CharacterBible,
        ))
        .expect("serialize reference source kind");
        let note = serde_json::to_string(&ContextSourceKind::Note).expect("serialize note source");

        assert_eq!(guide, r#"{"source_type":"guide","source_kind":"prose"}"#);
        assert_eq!(
            reference,
            r#"{"source_type":"reference","source_kind":"character_bible"}"#
        );
        assert_eq!(note, r#"{"source_type":"note"}"#);
    }

    #[test]
    fn analysis_defaults_to_guides_and_references_but_not_notes() {
        assert!(context_source_allowed_by_default(
            ConversationMode::Analysis,
            &ContextSourceKind::Guide(GuideKind::Critique)
        ));
        assert!(context_source_allowed_by_default(
            ConversationMode::Analysis,
            &ContextSourceKind::Reference(ReferenceKind::WorldSummary)
        ));
        assert!(!context_source_allowed_by_default(
            ConversationMode::Analysis,
            &ContextSourceKind::Note
        ));
    }

    #[test]
    fn editing_defaults_to_guides_and_canon_like_references() {
        assert!(context_source_allowed_by_default(
            ConversationMode::Editing,
            &ContextSourceKind::Guide(GuideKind::Prose)
        ));
        assert!(context_source_allowed_by_default(
            ConversationMode::Editing,
            &ContextSourceKind::Guide(GuideKind::Rewrite)
        ));
        assert!(context_source_allowed_by_default(
            ConversationMode::Editing,
            &ContextSourceKind::Reference(ReferenceKind::Terminology)
        ));
        assert!(!context_source_allowed_by_default(
            ConversationMode::Editing,
            &ContextSourceKind::Reference(ReferenceKind::Research)
        ));
    }

    #[test]
    fn ideation_defaults_to_guides_and_story_world_references() {
        assert!(context_source_allowed_by_default(
            ConversationMode::Ideation,
            &ContextSourceKind::Guide(GuideKind::Style)
        ));
        assert!(context_source_allowed_by_default(
            ConversationMode::Ideation,
            &ContextSourceKind::Reference(ReferenceKind::StorySummary)
        ));
        assert!(!context_source_allowed_by_default(
            ConversationMode::Ideation,
            &ContextSourceKind::Reference(ReferenceKind::Terminology)
        ));
        assert!(!context_source_allowed_by_default(
            ConversationMode::Ideation,
            &ContextSourceKind::Note
        ));
    }

    #[test]
    fn default_context_inclusion_respects_activation_and_review_state() {
        let approved_pinned_guide = ContextSource {
            path: "guides/prose.md".to_string(),
            kind: ContextSourceKind::Guide(GuideKind::Prose),
            activation_policy: ContextSourceActivationPolicy::Pinned,
            review_state: ContextSourceReviewState::Approved,
        };
        let explicit_only_guide = ContextSource {
            activation_policy: ContextSourceActivationPolicy::ExplicitOnly,
            ..approved_pinned_guide.clone()
        };
        let stale_guide = ContextSource {
            review_state: ContextSourceReviewState::Stale,
            ..approved_pinned_guide.clone()
        };

        assert!(context_source_included_by_default(
            ConversationMode::Editing,
            &approved_pinned_guide
        ));
        assert!(!context_source_included_by_default(
            ConversationMode::Editing,
            &explicit_only_guide
        ));
        assert!(!context_source_included_by_default(
            ConversationMode::Editing,
            &stale_guide
        ));
    }

    #[test]
    fn explicit_context_source_classification_overrides_filename_guessing() {
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Reference,
                "reference/ambiguous.md",
                Some(ContextSourceKind::Guide(GuideKind::Rewrite))
            ),
            Some(ContextSourceKind::Guide(GuideKind::Rewrite))
        );
    }

    #[test]
    fn confidently_named_reference_documents_classify_as_guides_or_references() {
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Reference,
                "guides/prose-guideline.md",
                None
            ),
            Some(ContextSourceKind::Guide(GuideKind::Prose))
        );
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Reference,
                "reference/world-summary.md",
                None
            ),
            Some(ContextSourceKind::Reference(ReferenceKind::WorldSummary))
        );
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Reference,
                "reference/character-bible.md",
                None
            ),
            Some(ContextSourceKind::Reference(ReferenceKind::CharacterBible))
        );
    }

    #[test]
    fn ambiguous_reference_documents_remain_unclassified() {
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Reference,
                "reference/brainstorm.md",
                None
            ),
            None
        );
        assert_eq!(
            classify_context_source_kind(
                DocumentType::Manuscript,
                "chapters/chapter-1.md",
                None
            ),
            None
        );
    }

    #[test]
    fn note_documents_remain_notes_without_aggressive_guessing() {
        assert_eq!(
            classify_context_source_kind(DocumentType::Note, "notes/scratch.md", None),
            Some(ContextSourceKind::Note)
        );
    }
}
