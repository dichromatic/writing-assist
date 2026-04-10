use writing_assist_core::{
    AnalysisComment, ContextSource, ConversationMode, DraftChange, IdeaCard, SelectionTarget,
    TaskContractError, TaskOutput, TaskRequest, TaskResult, TaskType,
};

use crate::{select_task_context, TaskContextSelectionRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicTaskRunRequest {
    pub mode: ConversationMode,
    pub task_type: TaskType,
    pub target: SelectionTarget,
    pub available_sources: Vec<ContextSource>,
    pub explicitly_selected_source_paths: Vec<String>,
}

pub fn run_deterministic_task(
    request: DeterministicTaskRunRequest,
) -> Result<TaskResult, TaskContractError> {
    let context = select_task_context(TaskContextSelectionRequest {
        mode: request.mode.clone(),
        target: request.target.clone(),
        available_sources: request.available_sources,
        explicitly_selected_source_paths: request.explicitly_selected_source_paths,
    });
    let task_request = TaskRequest::new(
        request.mode.clone(),
        request.task_type,
        request.target.clone(),
        context,
    );

    // Phase 2.5 is a deterministic provider-free stub, so outputs are structural placeholders only.
    let outputs = match request.mode {
        ConversationMode::Analysis => vec![TaskOutput::AnalysisComment(AnalysisComment::new(
            request.target,
            "Deterministic analysis placeholder for the selected text.",
        ))],
        ConversationMode::Editing => vec![TaskOutput::DraftChange(DraftChange::new(
            request.target.clone(),
            request.target.selected_text.clone(),
            format!("{} [deterministic edit]", request.target.selected_text),
        ))],
        ConversationMode::Ideation => vec![TaskOutput::IdeaCard(IdeaCard::new(
            "Deterministic idea",
            "Placeholder idea tied to the current task target.",
        ))],
    };

    TaskResult::new(&task_request, outputs)
}

#[cfg(test)]
mod tests {
    use writing_assist_core::{
        ContextSource, ContextSourceActivationPolicy, ContextSourceKind,
        ContextSourceReviewState, ConversationMode, GuideKind, ReferenceKind, SelectionTarget,
        TargetAnchor, TaskOutput, TaskType,
    };

    use super::{run_deterministic_task, DeterministicTaskRunRequest};

    fn selection_target() -> SelectionTarget {
        SelectionTarget::new(
            "chapters/chapter-1.md",
            "Selected paragraph.",
            10,
            29,
            vec![TargetAnchor::span(2)],
        )
    }

    fn source(
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

    fn request(mode: ConversationMode, task_type: TaskType) -> DeterministicTaskRunRequest {
        DeterministicTaskRunRequest {
            mode,
            task_type,
            target: selection_target(),
            available_sources: Vec::new(),
            explicitly_selected_source_paths: Vec::new(),
        }
    }

    #[test]
    fn analysis_returns_a_mode_correct_comment_without_draft_changes() {
        let result = run_deterministic_task(request(
            ConversationMode::Analysis,
            TaskType::AnalyzeSelection,
        ))
        .expect("analysis task should run");

        assert_eq!(result.mode, ConversationMode::Analysis);
        assert_eq!(result.outputs.len(), 1);
        assert!(matches!(
            result.outputs[0],
            TaskOutput::AnalysisComment(_)
        ));
    }

    #[test]
    fn editing_returns_a_bounded_draft_change_for_the_selected_text() {
        let target = selection_target();

        let result = run_deterministic_task(DeterministicTaskRunRequest {
            mode: ConversationMode::Editing,
            task_type: TaskType::RewriteSelection,
            target: target.clone(),
            available_sources: Vec::new(),
            explicitly_selected_source_paths: Vec::new(),
        })
        .expect("editing task should run");

        assert_eq!(result.mode, ConversationMode::Editing);
        assert_eq!(result.outputs.len(), 1);

        let TaskOutput::DraftChange(draft_change) = &result.outputs[0] else {
            panic!("editing task should return a draft change");
        };

        assert_eq!(draft_change.target, target);
        assert_eq!(draft_change.original_text, "Selected paragraph.");
        assert_ne!(draft_change.proposed_text, draft_change.original_text);
    }

    #[test]
    fn ideation_returns_an_idea_card_without_draft_changes() {
        let result = run_deterministic_task(request(
            ConversationMode::Ideation,
            TaskType::IdeateSelection,
        ))
        .expect("ideation task should run");

        assert_eq!(result.mode, ConversationMode::Ideation);
        assert_eq!(result.outputs.len(), 1);
        assert!(matches!(result.outputs[0], TaskOutput::IdeaCard(_)));
    }

    #[test]
    fn task_context_selection_is_used_before_result_creation() {
        let guide = source(
            "guides/prose.md",
            ContextSourceKind::Guide(GuideKind::Prose),
            ContextSourceActivationPolicy::Pinned,
            ContextSourceReviewState::Approved,
        );
        let research_reference = source(
            "research/ships.md",
            ContextSourceKind::Reference(ReferenceKind::Research),
            ContextSourceActivationPolicy::Pinned,
            ContextSourceReviewState::Approved,
        );

        let result = run_deterministic_task(DeterministicTaskRunRequest {
            mode: ConversationMode::Editing,
            task_type: TaskType::RewriteSelection,
            target: selection_target(),
            available_sources: vec![guide, research_reference],
            explicitly_selected_source_paths: Vec::new(),
        })
        .expect("editing task should run through task context selection");

        assert_eq!(result.mode, ConversationMode::Editing);
        assert!(matches!(result.outputs[0], TaskOutput::DraftChange(_)));
    }
}
