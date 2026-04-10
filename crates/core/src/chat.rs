use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ConversationMode, SelectionTarget};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "scope_type", content = "scope", rename_all = "snake_case")]
pub enum ChatThreadScope {
    Selection(SelectionTarget),
    Document { document_path: String },
    Project,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatMessageAuthor {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub id: Uuid,
    pub author: ChatMessageAuthor,
    pub content: String,
    pub created_at_ms: u64,
}

impl ChatMessage {
    pub fn new(author: ChatMessageAuthor, content: impl Into<String>, created_at_ms: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            author,
            content: content.into(),
            created_at_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatThread {
    pub id: Uuid,
    pub mode: ConversationMode,
    pub scope: ChatThreadScope,
    pub selected_context_source_paths: Vec<String>,
    pub messages: Vec<ChatMessage>,
}

impl ChatThread {
    pub fn new(
        mode: ConversationMode,
        scope: ChatThreadScope,
        selected_context_source_paths: Vec<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            mode,
            scope,
            selected_context_source_paths,
            messages: Vec::new(),
        }
    }

    pub fn set_scope(&mut self, scope: ChatThreadScope) {
        self.scope = scope;
    }

    pub fn add_message(
        &mut self,
        author: ChatMessageAuthor,
        content: impl Into<String>,
        created_at_ms: u64,
    ) -> Uuid {
        let message = ChatMessage::new(author, content, created_at_ms);
        let message_id = message.id;

        self.messages.push(message);

        message_id
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{ConversationMode, SelectionTarget, TargetAnchor};

    use super::{ChatMessageAuthor, ChatThread, ChatThreadScope};

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
    fn new_thread_stores_mode_scope_and_context_source_paths() {
        let scope = ChatThreadScope::Selection(selection_target());
        let selected_context_source_paths = vec![
            "guides/prose.md".to_string(),
            "world/story-summary.md".to_string(),
        ];

        let thread = ChatThread::new(
            ConversationMode::Analysis,
            scope.clone(),
            selected_context_source_paths.clone(),
        );

        assert_ne!(thread.id, Uuid::nil());
        assert_eq!(thread.mode, ConversationMode::Analysis);
        assert_eq!(thread.scope, scope);
        assert_eq!(
            thread.selected_context_source_paths,
            selected_context_source_paths
        );
        assert_eq!(thread.messages, Vec::new());
    }

    #[test]
    fn thread_scope_can_change_without_silently_changing_mode() {
        let mut thread = ChatThread::new(
            ConversationMode::Editing,
            ChatThreadScope::Selection(selection_target()),
            Vec::new(),
        );

        thread.set_scope(ChatThreadScope::Document {
            document_path: "chapters/chapter-2.md".to_string(),
        });

        assert_eq!(thread.mode, ConversationMode::Editing);
        assert_eq!(
            thread.scope,
            ChatThreadScope::Document {
                document_path: "chapters/chapter-2.md".to_string(),
            }
        );
    }

    #[test]
    fn selected_context_source_paths_are_independent_of_messages() {
        let mut thread = ChatThread::new(
            ConversationMode::Ideation,
            ChatThreadScope::Project,
            vec!["notes/seed.md".to_string()],
        );

        thread.add_message(ChatMessageAuthor::User, "Give me three options.", 1000);

        assert_eq!(
            thread.selected_context_source_paths,
            vec!["notes/seed.md".to_string()]
        );
        assert_eq!(thread.messages.len(), 1);
    }

    #[test]
    fn chat_messages_append_in_order_with_author_and_timestamp() {
        let mut thread = ChatThread::new(
            ConversationMode::Analysis,
            ChatThreadScope::Selection(selection_target()),
            Vec::new(),
        );

        let first_id = thread.add_message(ChatMessageAuthor::User, "What is weak here?", 1000);
        let second_id =
            thread.add_message(ChatMessageAuthor::Assistant, "The transition is abrupt.", 1500);

        assert_ne!(first_id, Uuid::nil());
        assert_ne!(second_id, Uuid::nil());
        assert_ne!(first_id, second_id);
        assert_eq!(thread.messages.len(), 2);
        assert_eq!(thread.messages[0].author, ChatMessageAuthor::User);
        assert_eq!(thread.messages[0].content, "What is weak here?");
        assert_eq!(thread.messages[0].created_at_ms, 1000);
        assert_eq!(thread.messages[1].author, ChatMessageAuthor::Assistant);
        assert_eq!(thread.messages[1].content, "The transition is abrupt.");
        assert_eq!(thread.messages[1].created_at_ms, 1500);
    }
}
