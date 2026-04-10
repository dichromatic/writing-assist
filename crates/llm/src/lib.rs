use anyhow::Result;
use serde::{Deserialize, Serialize};
use writing_assist_core::ConversationMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub mode: ConversationMode,
    pub prompt: String,
}

pub trait TaskExecutor {
    fn execute(&self, request: TaskRequest) -> Result<String>;
}
