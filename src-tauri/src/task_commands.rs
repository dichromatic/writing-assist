use serde::{Deserialize, Serialize};
use writing_assist_core::{ContextSource, ConversationMode, SelectionTarget, TaskResult, TaskType};
use writing_assist_orchestrator::{run_deterministic_task, DeterministicTaskRunRequest};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeterministicTaskCommandRequest {
    pub mode: ConversationMode,
    pub task_type: TaskType,
    pub target: SelectionTarget,
    pub available_sources: Vec<ContextSource>,
    pub explicitly_selected_source_paths: Vec<String>,
}

pub fn run_deterministic_task_command(
    request: DeterministicTaskCommandRequest,
) -> Result<TaskResult, String> {
    // Keep the Tauri command boundary thin: deserialize input, delegate to orchestrator, format errors.
    run_deterministic_task(DeterministicTaskRunRequest {
        mode: request.mode,
        task_type: request.task_type,
        target: request.target,
        available_sources: request.available_sources,
        explicitly_selected_source_paths: request.explicitly_selected_source_paths,
    })
    .map_err(|error| format!("Failed to run deterministic task: {error}"))
}

#[cfg(test)]
mod tests {
    use writing_assist_core::{
        ContextSource, ContextSourceActivationPolicy, ContextSourceKind,
        ContextSourceReviewState, ConversationMode, GuideKind, SelectionTarget, TargetAnchor,
        TaskOutput, TaskType,
    };

    use super::{run_deterministic_task_command, DeterministicTaskCommandRequest};

    fn selection_target() -> SelectionTarget {
        SelectionTarget::new(
            "chapters/chapter-1.md",
            "Selected paragraph.",
            10,
            29,
            vec![TargetAnchor::span(2)],
        )
    }

    #[test]
    fn maps_command_input_to_the_orchestrator_task_runner() {
        let result = run_deterministic_task_command(DeterministicTaskCommandRequest {
            mode: ConversationMode::Analysis,
            task_type: TaskType::AnalyzeSelection,
            target: selection_target(),
            available_sources: vec![ContextSource {
                path: "guides/prose.md".to_string(),
                kind: ContextSourceKind::Guide(GuideKind::Prose),
                activation_policy: ContextSourceActivationPolicy::Pinned,
                review_state: ContextSourceReviewState::Approved,
            }],
            explicitly_selected_source_paths: Vec::new(),
        })
        .expect("command-adjacent function should run deterministic task");

        assert_eq!(result.mode, ConversationMode::Analysis);
        assert!(matches!(result.outputs[0], TaskOutput::AnalysisComment(_)));
    }
}
