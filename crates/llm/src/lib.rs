use anyhow::Result;
use serde::{Deserialize, Serialize};
use writing_assist_core::ConversationMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassRequest {
    pub mode: ConversationMode,
    pub prompt: String,
}

pub trait PassExecutor {
    fn execute(&self, request: PassRequest) -> Result<String>;
}
