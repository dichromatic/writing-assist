use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::{context_source_included_by_default, ContextSource, ConversationMode};

pub const TASK_CONTRACT_SCHEMA_VERSION: u16 = 1;

/// High-level operation the orchestrator is being asked to perform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    AnalyzeSelection,
    RewriteSelection,
    IdeateSelection,
    Chat,
}

/// Logical target category for a selection. Sections/scenes/windows are task targets,
/// not necessarily raw parser-emitted spans.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetAnchorKind {
    Span,
    Section,
    Scene,
    Window,
}

/// Stable reference to the parsed structure that overlaps the selected text.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetAnchor {
    pub kind: TargetAnchorKind,
    pub ordinal: usize,
}

impl TargetAnchor {
    pub fn span(ordinal: usize) -> Self {
        Self {
            kind: TargetAnchorKind::Span,
            ordinal,
        }
    }

    pub fn section(ordinal: usize) -> Self {
        Self {
            kind: TargetAnchorKind::Section,
            ordinal,
        }
    }

    pub fn scene(ordinal: usize) -> Self {
        Self {
            kind: TargetAnchorKind::Scene,
            ordinal,
        }
    }

    pub fn window(ordinal: usize) -> Self {
        Self {
            kind: TargetAnchorKind::Window,
            ordinal,
        }
    }
}

/// Text range and parsed anchors a task is allowed to inspect or modify.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectionTarget {
    pub document_path: String,
    pub selected_text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub anchors: Vec<TargetAnchor>,
}

impl SelectionTarget {
    pub fn new(
        document_path: impl Into<String>,
        selected_text: impl Into<String>,
        start_char: usize,
        end_char: usize,
        anchors: Vec<TargetAnchor>,
    ) -> Self {
        Self {
            document_path: document_path.into(),
            selected_text: selected_text.into(),
            start_char,
            end_char,
            anchors,
        }
    }

    pub fn span_ordinals(&self) -> Vec<usize> {
        self.ordinals_for_kind(TargetAnchorKind::Span)
    }

    pub fn section_ordinals(&self) -> Vec<usize> {
        self.ordinals_for_kind(TargetAnchorKind::Section)
    }

    pub fn scene_ordinals(&self) -> Vec<usize> {
        self.ordinals_for_kind(TargetAnchorKind::Scene)
    }

    pub fn window_ordinals(&self) -> Vec<usize> {
        self.ordinals_for_kind(TargetAnchorKind::Window)
    }

    fn contains_target(&self, other: &SelectionTarget) -> bool {
        self.document_path == other.document_path
            && other.start_char >= self.start_char
            && other.end_char <= self.end_char
    }

    fn ordinals_for_kind(&self, kind: TargetAnchorKind) -> Vec<usize> {
        self.anchors
            .iter()
            .filter(|anchor| anchor.kind == kind)
            .map(|anchor| anchor.ordinal)
            .collect()
    }
}

/// Deterministic context selected for a task before retrieval or provider code exists.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextBundle {
    pub target: Option<SelectionTarget>,
    pub included_sources: Vec<ContextSource>,
    pub excluded_sources: Vec<ContextSource>,
}

impl ContextBundle {
    pub fn empty() -> Self {
        Self {
            target: None,
            included_sources: Vec::new(),
            excluded_sources: Vec::new(),
        }
    }

    pub fn from_sources(
        mode: ConversationMode,
        target: SelectionTarget,
        sources: Vec<ContextSource>,
        explicitly_selected_paths: &[String],
    ) -> Self {
        let explicit_paths: HashSet<_> =
            explicitly_selected_paths.iter().map(String::as_str).collect();
        let mut included_sources = Vec::new();
        let mut excluded_sources = Vec::new();

        for source in sources {
            let explicitly_selected = explicit_paths.contains(source.path.as_str());
            // Explicit user selection can bring notes into scope, but stale/pending material still stays out.
            let include = if explicitly_selected {
                matches!(
                    source.review_state,
                    crate::ContextSourceReviewState::UserAuthored
                        | crate::ContextSourceReviewState::Approved
                )
            } else {
                context_source_included_by_default(mode.clone(), &source)
            };

            if include {
                included_sources.push(source);
            } else {
                excluded_sources.push(source);
            }
        }

        Self {
            target: Some(target),
            included_sources,
            excluded_sources,
        }
    }
}

/// Provider-agnostic request contract shared by chat, orchestrator, and later LLM adapters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskRequest {
    pub id: Uuid,
    pub schema_version: u16,
    pub mode: ConversationMode,
    pub task_type: TaskType,
    pub target: SelectionTarget,
    pub context: ContextBundle,
}

impl TaskRequest {
    pub fn new(
        mode: ConversationMode,
        task_type: TaskType,
        target: SelectionTarget,
        mut context: ContextBundle,
    ) -> Self {
        // Keep request/context target state aligned even when callers start from `ContextBundle::empty`.
        context.target = Some(target.clone());

        Self {
            id: Uuid::new_v4(),
            schema_version: TASK_CONTRACT_SCHEMA_VERSION,
            mode,
            task_type,
            target,
            context,
        }
    }
}

/// Non-mutating critique or observation attached to a target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnalysisComment {
    pub target: SelectionTarget,
    pub message: String,
}

impl AnalysisComment {
    pub fn new(target: SelectionTarget, message: impl Into<String>) -> Self {
        Self {
            target,
            message: message.into(),
        }
    }
}

/// Proposed edit that must stay bounded to the originating editing target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DraftChange {
    pub target: SelectionTarget,
    pub original_text: String,
    pub proposed_text: String,
}

impl DraftChange {
    pub fn new(
        target: SelectionTarget,
        original_text: impl Into<String>,
        proposed_text: impl Into<String>,
    ) -> Self {
        Self {
            target,
            original_text: original_text.into(),
            proposed_text: proposed_text.into(),
        }
    }
}

/// Ideation-only structured output that does not directly modify manuscript text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdeaCard {
    pub title: String,
    pub body: String,
}

impl IdeaCard {
    pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
        }
    }
}

/// Structured result item emitted by a mode-aware task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "output_type", content = "output", rename_all = "snake_case")]
pub enum TaskOutput {
    AnalysisComment(AnalysisComment),
    DraftChange(DraftChange),
    IdeaCard(IdeaCard),
}

/// Versioned result envelope tied back to a single task request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskResult {
    pub request_id: Uuid,
    pub schema_version: u16,
    pub mode: ConversationMode,
    pub outputs: Vec<TaskOutput>,
}

impl TaskResult {
    pub fn new(
        request: &TaskRequest,
        outputs: Vec<TaskOutput>,
    ) -> Result<Self, TaskContractError> {
        validate_outputs_for_mode(request, &outputs)?;

        Ok(Self {
            request_id: request.id,
            schema_version: TASK_CONTRACT_SCHEMA_VERSION,
            mode: request.mode.clone(),
            outputs,
        })
    }
}

/// Contract-level validation failures before draft changes reach persistence.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TaskContractError {
    #[error("draft changes are not allowed in {mode:?} mode")]
    DraftChangesNotAllowed { mode: ConversationMode },
    #[error("draft change target is outside the task request target")]
    DraftChangeOutsideTarget,
}

fn validate_outputs_for_mode(
    request: &TaskRequest,
    outputs: &[TaskOutput],
) -> Result<(), TaskContractError> {
    for output in outputs {
        if let TaskOutput::DraftChange(draft_change) = output {
            if request.mode != ConversationMode::Editing {
                return Err(TaskContractError::DraftChangesNotAllowed {
                    mode: request.mode.clone(),
                });
            }

            if !request.target.contains_target(&draft_change.target) {
                return Err(TaskContractError::DraftChangeOutsideTarget);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{
        ContextSource, ContextSourceActivationPolicy, ContextSourceKind,
        ContextSourceReviewState, ConversationMode, GuideKind, ReferenceKind,
    };

    use super::{
        AnalysisComment, ContextBundle, DraftChange, IdeaCard, TaskContractError, TaskOutput,
        TaskRequest, TaskResult, TaskType, SelectionTarget, TargetAnchor,
        TASK_CONTRACT_SCHEMA_VERSION,
    };

    fn selection_target() -> SelectionTarget {
        SelectionTarget::new(
            "chapters/chapter-1.md",
            "Selected paragraph.",
            10,
            29,
            vec![TargetAnchor::span(2), TargetAnchor::section(1)],
        )
    }

    fn context_source(
        path: &str,
        kind: ContextSourceKind,
        activation_policy: ContextSourceActivationPolicy,
        review_state: ContextSourceReviewState,
    ) -> ContextSource {
        ContextSource {
            path: path.to_string(),
            kind,
            activation_policy,
            review_state,
        }
    }

    #[test]
    fn task_request_has_stable_schema_version_and_task_id() {
        let target = selection_target();
        let request = TaskRequest::new(
            ConversationMode::Analysis,
            TaskType::AnalyzeSelection,
            target.clone(),
            ContextBundle::empty(),
        );

        assert_eq!(request.schema_version, TASK_CONTRACT_SCHEMA_VERSION);
        assert_ne!(request.id, Uuid::nil());
        assert_eq!(request.context.target, Some(target));
    }

    #[test]
    fn selection_target_can_reference_span_section_scene_and_window_targets() {
        let target = SelectionTarget::new(
            "chapters/chapter-1.md",
            "Selected passage.",
            5,
            22,
            vec![
                TargetAnchor::span(1),
                TargetAnchor::section(2),
                TargetAnchor::scene(3),
                TargetAnchor::window(4),
            ],
        );

        assert_eq!(target.span_ordinals(), vec![1]);
        assert_eq!(target.section_ordinals(), vec![2]);
        assert_eq!(target.scene_ordinals(), vec![3]);
        assert_eq!(target.window_ordinals(), vec![4]);
    }

    #[test]
    fn context_bundle_uses_default_policy_and_allows_explicit_notes() {
        let guide = context_source(
            "guides/prose.md",
            ContextSourceKind::Guide(GuideKind::Prose),
            ContextSourceActivationPolicy::Pinned,
            ContextSourceReviewState::Approved,
        );
        let note = context_source(
            "notes/scratch.md",
            ContextSourceKind::Note,
            ContextSourceActivationPolicy::ExplicitOnly,
            ContextSourceReviewState::UserAuthored,
        );
        let stale_reference = context_source(
            "world/stale.md",
            ContextSourceKind::Reference(ReferenceKind::WorldSummary),
            ContextSourceActivationPolicy::Pinned,
            ContextSourceReviewState::Stale,
        );

        let target = selection_target();
        let bundle = ContextBundle::from_sources(
            ConversationMode::Editing,
            target.clone(),
            vec![guide.clone(), note.clone(), stale_reference.clone()],
            &["notes/scratch.md".to_string()],
        );

        assert_eq!(bundle.target, Some(target));
        assert_eq!(bundle.included_sources, vec![guide, note]);
        assert_eq!(bundle.excluded_sources, vec![stale_reference]);
    }

    #[test]
    fn notes_are_excluded_from_context_by_default() {
        let note = context_source(
            "notes/scratch.md",
            ContextSourceKind::Note,
            ContextSourceActivationPolicy::Retrieved,
            ContextSourceReviewState::UserAuthored,
        );

        let bundle = ContextBundle::from_sources(
            ConversationMode::Analysis,
            selection_target(),
            vec![note.clone()],
            &[],
        );

        assert_eq!(bundle.included_sources, Vec::new());
        assert_eq!(bundle.excluded_sources, vec![note]);
    }

    #[test]
    fn analysis_cannot_emit_draft_changes() {
        let request = TaskRequest::new(
            ConversationMode::Analysis,
            TaskType::AnalyzeSelection,
            selection_target(),
            ContextBundle::empty(),
        );
        let draft_change = DraftChange::new(selection_target(), "old", "new");

        let result = TaskResult::new(
            &request,
            vec![TaskOutput::DraftChange(draft_change)],
        );

        assert_eq!(
            result,
            Err(TaskContractError::DraftChangesNotAllowed {
                mode: ConversationMode::Analysis,
            })
        );
    }

    #[test]
    fn editing_accepts_bounded_draft_changes() {
        let request = TaskRequest::new(
            ConversationMode::Editing,
            TaskType::RewriteSelection,
            selection_target(),
            ContextBundle::empty(),
        );
        let draft_change = DraftChange::new(selection_target(), "Selected paragraph.", "Rewritten paragraph.");

        let result = TaskResult::new(
            &request,
            vec![TaskOutput::DraftChange(draft_change.clone())],
        )
        .expect("bounded editing output should be valid");

        assert_eq!(result.schema_version, TASK_CONTRACT_SCHEMA_VERSION);
        assert_eq!(result.request_id, request.id);
        assert_eq!(result.outputs, vec![TaskOutput::DraftChange(draft_change)]);
    }

    #[test]
    fn editing_rejects_draft_changes_outside_the_request_target() {
        let request = TaskRequest::new(
            ConversationMode::Editing,
            TaskType::RewriteSelection,
            selection_target(),
            ContextBundle::empty(),
        );
        let outside_target = SelectionTarget::new(
            "chapters/chapter-1.md",
            "Outside paragraph.",
            40,
            58,
            vec![TargetAnchor::span(4)],
        );

        let result = TaskResult::new(
            &request,
            vec![TaskOutput::DraftChange(DraftChange::new(outside_target, "old", "new"))],
        );

        assert_eq!(result, Err(TaskContractError::DraftChangeOutsideTarget));
    }

    #[test]
    fn ideation_accepts_idea_cards_but_not_draft_changes() {
        let request = TaskRequest::new(
            ConversationMode::Ideation,
            TaskType::IdeateSelection,
            selection_target(),
            ContextBundle::empty(),
        );

        let idea_result = TaskResult::new(
            &request,
            vec![TaskOutput::IdeaCard(IdeaCard::new("Option A", "Try a quieter reversal."))],
        );
        assert!(idea_result.is_ok());

        let draft_result = TaskResult::new(
            &request,
            vec![TaskOutput::DraftChange(DraftChange::new(selection_target(), "old", "new"))],
        );
        assert_eq!(
            draft_result,
            Err(TaskContractError::DraftChangesNotAllowed {
                mode: ConversationMode::Ideation,
            })
        );
    }

    #[test]
    fn analysis_accepts_analysis_comments() {
        let request = TaskRequest::new(
            ConversationMode::Analysis,
            TaskType::AnalyzeSelection,
            selection_target(),
            ContextBundle::empty(),
        );

        let result = TaskResult::new(
            &request,
            vec![TaskOutput::AnalysisComment(AnalysisComment::new(
                selection_target(),
                "The paragraph shifts point of view.",
            ))],
        );

        assert!(result.is_ok());
    }
}
