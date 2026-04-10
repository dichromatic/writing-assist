use writing_assist_core::{ContextBundle, ContextSource, ConversationMode, SelectionTarget};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskContextSelectionRequest {
    pub mode: ConversationMode,
    pub target: SelectionTarget,
    pub available_sources: Vec<ContextSource>,
    pub explicitly_selected_source_paths: Vec<String>,
}

pub fn select_task_context(request: TaskContextSelectionRequest) -> ContextBundle {
    // Keep Phase 2.3 orchestration deterministic and policy-compatible by delegating inclusion rules to core.
    ContextBundle::from_sources(
        request.mode,
        request.target,
        request.available_sources,
        &request.explicitly_selected_source_paths,
    )
}

#[cfg(test)]
mod tests {
    use writing_assist_core::{
        ContextSource, ContextSourceActivationPolicy, ContextSourceKind,
        ContextSourceReviewState, ConversationMode, GuideKind, ReferenceKind, SelectionTarget,
        TargetAnchor,
    };

    use super::{select_task_context, TaskContextSelectionRequest};

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

    #[test]
    fn selects_task_context_with_the_selected_target() {
        let target = selection_target();

        let bundle = select_task_context(TaskContextSelectionRequest {
            mode: ConversationMode::Analysis,
            target: target.clone(),
            available_sources: Vec::new(),
            explicitly_selected_source_paths: Vec::new(),
        });

        assert_eq!(bundle.target, Some(target));
        assert_eq!(bundle.included_sources, Vec::new());
        assert_eq!(bundle.excluded_sources, Vec::new());
    }

    #[test]
    fn applies_core_mode_policy_to_available_sources() {
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

        let bundle = select_task_context(TaskContextSelectionRequest {
            mode: ConversationMode::Editing,
            target: selection_target(),
            available_sources: vec![guide.clone(), research_reference.clone()],
            explicitly_selected_source_paths: Vec::new(),
        });

        assert_eq!(bundle.included_sources, vec![guide]);
        assert_eq!(bundle.excluded_sources, vec![research_reference]);
    }

    #[test]
    fn preserves_excluded_sources_for_later_context_inspection() {
        let note = source(
            "notes/scratch.md",
            ContextSourceKind::Note,
            ContextSourceActivationPolicy::Retrieved,
            ContextSourceReviewState::UserAuthored,
        );

        let bundle = select_task_context(TaskContextSelectionRequest {
            mode: ConversationMode::Analysis,
            target: selection_target(),
            available_sources: vec![note.clone()],
            explicitly_selected_source_paths: Vec::new(),
        });

        assert_eq!(bundle.included_sources, Vec::new());
        assert_eq!(bundle.excluded_sources, vec![note]);
    }

    #[test]
    fn allows_explicit_user_authored_notes_but_keeps_stale_notes_excluded() {
        let user_note = source(
            "notes/scratch.md",
            ContextSourceKind::Note,
            ContextSourceActivationPolicy::ExplicitOnly,
            ContextSourceReviewState::UserAuthored,
        );
        let stale_note = source(
            "notes/old.md",
            ContextSourceKind::Note,
            ContextSourceActivationPolicy::ExplicitOnly,
            ContextSourceReviewState::Stale,
        );

        let bundle = select_task_context(TaskContextSelectionRequest {
            mode: ConversationMode::Ideation,
            target: selection_target(),
            available_sources: vec![user_note.clone(), stale_note.clone()],
            explicitly_selected_source_paths: vec![user_note.path.clone(), stale_note.path.clone()],
        });

        assert_eq!(bundle.included_sources, vec![user_note]);
        assert_eq!(bundle.excluded_sources, vec![stale_note]);
    }
}
