use writing_assist_core::ConversationMode;

pub fn default_scope(mode: ConversationMode) -> &'static str {
    match mode {
        ConversationMode::Analysis => "selection",
        ConversationMode::Editing => "selection",
        ConversationMode::Ideation => "selection",
    }
}
