use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use crate::SemanticPassLabPacket;

const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "moonshotai/kimi-k2-instruct-0905";
const DEFAULT_TIMEOUT_SECS: u64 = 180;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 20;
const DEFAULT_MAX_RETRIES: u32 = 2;
const DEFAULT_MAX_TOKENS: u32 = 2_000;

/// Lab-only NIM client configuration.
///
/// This stays inside the experimental tool so provider wiring does not leak
/// into production crates before the semantic contract stabilizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NimLabConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticPassLabResponse {
    pub model: String,
    pub raw_content: String,
    pub parsed_output: Value,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    nvext: Value,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChatCompletionResponse {
    model: Option<String>,
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct ChatChoiceMessage {
    content: String,
}

impl NimLabConfig {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("NIM_API_KEY")
            .context("missing NIM_API_KEY environment variable for semantic-pass lab")?;
        let base_url =
            std::env::var("NIM_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let model = std::env::var("NIM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let timeout_secs = std::env::var("NIM_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let max_retries = std::env::var("NIM_MAX_RETRIES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_MAX_RETRIES);
        let max_tokens = std::env::var("NIM_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS);

        Ok(Self {
            api_key,
            base_url,
            model,
            timeout_secs,
            max_retries,
            max_tokens,
        })
    }
}

pub fn build_nim_request(
    config: &NimLabConfig,
    packet: &SemanticPassLabPacket,
) -> Result<Value> {
    let request = ChatCompletionRequest {
        model: &config.model,
        messages: vec![ChatMessage {
            role: "user",
            content: &packet.prompt,
        }],
        temperature: 0.1,
        stream: false,
        max_tokens: Some(config.max_tokens),
        // NVIDIA documents recommend guided_json instead of plain json_object.
        nvext: json!({
            "guided_json": crate::semantic_output_schema(packet.task_kind),
        }),
    };

    serde_json::to_value(request).context("nim request should serialize")
}

pub fn execute_semantic_pass(
    config: &NimLabConfig,
    packet: &SemanticPassLabPacket,
) -> Result<SemanticPassLabResponse> {
    let request_body = build_nim_request(config, packet)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(Duration::from_secs(config.timeout_secs))
        .build()
        .context("failed to build reqwest client for NIM")?;
    let response = send_with_retries(&client, config, &request_body)?;

    let parsed_response: ChatCompletionResponse = response
        .json()
        .context("failed to parse NIM chat completion response body")?;
    parse_semantic_pass_response(parsed_response)
}

fn send_with_retries(
    client: &Client,
    config: &NimLabConfig,
    request_body: &Value,
) -> Result<reqwest::blocking::Response> {
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match client
            .post(&config.base_url)
            .bearer_auth(&config.api_key)
            .json(request_body)
            .send()
        {
            Ok(response) => {
                return response
                    .error_for_status()
                    .context("NIM returned an error status for semantic-pass lab");
            }
            Err(error) if should_retry_request(&error, attempt, config.max_retries) => {
                last_error = Some(error);
                continue;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to call NIM endpoint {}", config.base_url));
            }
        }
    }

    Err(last_error
        .expect("retry loop should capture the final request error"))
    .with_context(|| {
        format!(
            "failed to call NIM endpoint {} after {} attempts",
            config.base_url,
            config.max_retries + 1
        )
    })
}

fn should_retry_request(error: &reqwest::Error, attempt: u32, max_retries: u32) -> bool {
    should_retry_flags(error.is_timeout(), error.is_connect(), attempt, max_retries)
}

fn should_retry_flags(
    is_timeout: bool,
    is_connect: bool,
    attempt: u32,
    max_retries: u32,
) -> bool {
    attempt < max_retries && (is_timeout || is_connect)
}

fn parse_semantic_pass_response(
    response: ChatCompletionResponse,
) -> Result<SemanticPassLabResponse> {
    let message = response
        .choices
        .first()
        .ok_or_else(|| anyhow!("NIM response did not include any choices"))?;
    let raw_content = message.message.content.trim().to_string();
    let parsed_output = parse_guided_json_content(&raw_content)
        .with_context(|| {
            format!(
                "semantic-pass lab expected JSON content from guided_json response; raw content starts with: {}",
                truncate_for_error(&raw_content, 160)
            )
        })?;

    Ok(SemanticPassLabResponse {
        model: response.model.unwrap_or_else(|| "unknown".to_string()),
        raw_content,
        parsed_output,
    })
}

fn parse_guided_json_content(content: &str) -> Result<Value> {
    let normalized = normalize_guided_json_content(content);
    let mut attempt_errors = Vec::new();

    match serde_json::from_str::<Value>(&normalized) {
        Ok(parsed) => return Ok(parsed),
        Err(error) => attempt_errors.push(format!("direct parse failed: {error}")),
    }

    if let Some(fenced) = extract_fenced_json_block(&normalized) {
        match serde_json::from_str::<Value>(&fenced) {
            Ok(parsed) => return Ok(parsed),
            Err(error) => attempt_errors.push(format!("fenced parse failed: {error}")),
        }
    }

    if let Some(candidate) = extract_first_json_object(&normalized) {
        match serde_json::from_str::<Value>(&candidate) {
            Ok(parsed) => return Ok(parsed),
            Err(error) => attempt_errors.push(format!("object extraction parse failed: {error}")),
        }
    }

    Err(anyhow!(
        "{} | leading_codepoints={} | normalized_starts_with={}",
        attempt_errors.join(" | "),
        leading_codepoints(&normalized, 8),
        truncate_for_error(&normalized, 160)
    ))
}

fn normalize_guided_json_content(content: &str) -> String {
    content
        .trim()
        .trim_start_matches(is_leading_json_noise)
        .to_string()
}

fn is_leading_json_noise(character: char) -> bool {
    matches!(
        character,
        '\u{feff}' | '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}'
    )
}

fn extract_fenced_json_block(content: &str) -> Option<String> {
    let trimmed = content.trim();
    let stripped = trimmed.strip_prefix("```json").or_else(|| trimmed.strip_prefix("```"))?;
    let without_trailing = stripped.trim().strip_suffix("```")?;
    Some(without_trailing.trim().to_string())
}

fn extract_first_json_object(content: &str) -> Option<String> {
    let start = content.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, character) in content[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }

            match character {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + index + character.len_utf8();
                    return Some(content[start..end].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

fn truncate_for_error(text: &str, limit: usize) -> String {
    let truncated = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn leading_codepoints(text: &str, limit: usize) -> String {
    text.chars()
        .take(limit)
        .map(|character| format!("U+{:04X}", character as u32))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use serde_json::Value;
    use writing_assist_core::{DocumentArchetype, DocumentType};

    use crate::{
        LabClusterRecord, LabDefinitionRecord, LabFieldRecord, LabSummarySeedRecord,
        SemanticBundleShape, SemanticPassLabPacket,
    };

    use super::{build_nim_request, NimLabConfig};

    #[test]
    fn nim_request_uses_guided_json_schema_and_model() {
        let config = NimLabConfig {
            api_key: "test-key".to_string(),
            base_url: "https://example.invalid/v1/chat/completions".to_string(),
            model: "moonshotai/kimi-k2-instruct-0905".to_string(),
            timeout_secs: 180,
            max_retries: 2,
            max_tokens: 2_000,
        };
        let packet = minimal_packet();

        let request = build_nim_request(&config, &packet).expect("request should build");

        assert_eq!(
            request.get("model").and_then(Value::as_str),
            Some("moonshotai/kimi-k2-instruct-0905")
        );
        assert_eq!(
            request
                .get("nvext")
                .and_then(|nvext| nvext.get("guided_json"))
                .and_then(|schema| schema.get("type"))
                .and_then(Value::as_str),
            Some("object")
        );
        assert_eq!(
            request.get("max_tokens").and_then(Value::as_u64),
            Some(2_000)
        );
    }

    #[test]
    fn env_config_uses_documented_defaults() {
        unsafe {
            std::env::set_var("NIM_API_KEY", "test-key");
            std::env::remove_var("NIM_MODEL");
            std::env::remove_var("NIM_BASE_URL");
            std::env::remove_var("NIM_TIMEOUT_SECS");
            std::env::remove_var("NIM_MAX_RETRIES");
            std::env::remove_var("NIM_MAX_TOKENS");
        }

        let config = NimLabConfig::from_env().expect("env config should load");

        assert_eq!(config.model, "moonshotai/kimi-k2-instruct-0905");
        assert_eq!(
            config.base_url,
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
        assert_eq!(config.timeout_secs, 180);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.max_tokens, 2_000);
    }

    #[test]
    fn semantic_response_parser_reads_guided_json_output() {
        let response = super::parse_semantic_pass_response(super::ChatCompletionResponse {
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            choices: vec![super::ChatChoice {
                message: super::ChatChoiceMessage {
                    content: json!({
                        "proposed_entities": [],
                        "proposed_relationships": [],
                        "proposed_terminology": [],
                        "rejected_evidence": [],
                        "open_questions": ["Ambiguous title surface"]
                    })
                    .to_string(),
                },
            }],
        })
        .expect("response should parse");

        assert_eq!(response.model, "moonshotai/kimi-k2-instruct-0905");
        assert_eq!(
            response
                .parsed_output
                .get("open_questions")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn semantic_response_parser_accepts_fenced_json_content() {
        let response = super::parse_semantic_pass_response(super::ChatCompletionResponse {
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            choices: vec![super::ChatChoice {
                message: super::ChatChoiceMessage {
                    content: "```json\n{\"proposed_entities\":[],\"proposed_relationships\":[],\"rejected_evidence\":[],\"open_questions\":[],\"candidate_entities_needing_review\":[]}\n```".to_string(),
                },
            }],
        })
        .expect("fenced guided-json response should parse");

        assert!(response.parsed_output.is_object());
    }

    #[test]
    fn semantic_response_parser_extracts_json_after_leading_text() {
        let response = super::parse_semantic_pass_response(super::ChatCompletionResponse {
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            choices: vec![super::ChatChoice {
                message: super::ChatChoiceMessage {
                    content: "Here is the requested JSON:\n{\"proposed_entities\":[],\"proposed_relationships\":[],\"rejected_evidence\":[],\"open_questions\":[],\"candidate_entities_needing_review\":[]}".to_string(),
                },
            }],
        })
        .expect("response with leading text should parse");

        assert!(response.parsed_output.is_object());
    }

    #[test]
    fn semantic_response_parser_strips_utf_bom_before_json() {
        let response = super::parse_semantic_pass_response(super::ChatCompletionResponse {
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            choices: vec![super::ChatChoice {
                message: super::ChatChoiceMessage {
                    content: "\u{feff}{\"proposed_entities\":[],\"proposed_relationships\":[],\"rejected_evidence\":[],\"open_questions\":[],\"candidate_entities_needing_review\":[]}".to_string(),
                },
            }],
        })
        .expect("response with BOM should parse");

        assert!(response.parsed_output.is_object());
    }

    #[test]
    fn semantic_response_parser_preserves_specific_json_error_context() {
        let error = super::parse_semantic_pass_response(super::ChatCompletionResponse {
            model: Some("moonshotai/kimi-k2-instruct-0905".to_string()),
            choices: vec![super::ChatChoice {
                message: super::ChatChoiceMessage {
                    content: "{\"proposed_entities\":[}".to_string(),
                },
            }],
        })
        .expect_err("malformed json should fail");

        let message = format!("{error:#}");
        assert!(message.contains("object extraction parse failed"));
        assert!(message.contains("leading_codepoints=U+007B"));
    }

    #[test]
    fn retry_policy_retries_only_for_timeout_like_failures() {
        assert!(super::should_retry_flags(true, false, 0, 2));
        assert!(super::should_retry_flags(false, true, 0, 2));
        assert!(!super::should_retry_flags(true, false, 2, 2));
        assert!(!super::should_retry_flags(false, false, 0, 2));
    }

    fn minimal_packet() -> SemanticPassLabPacket {
        SemanticPassLabPacket {
            experiment_name: "semantic-pass-lab-v1".to_string(),
            bundle_shape: SemanticBundleShape::Curated,
            task_kind: crate::SemanticPassTaskKind::ManuscriptEntityConsolidation,
            document_path: "1. Radiant Firth.md".to_string(),
            document_type: DocumentType::Manuscript,
            archetype: DocumentArchetype::Manuscript,
            prompt: "Return JSON".to_string(),
            output_contract: "{}".to_string(),
            clusters: vec![LabClusterRecord {
                id: "cluster-1".to_string(),
                display_surface: "Radiant Firth".to_string(),
                member_surfaces: vec!["Radiant Firth".to_string()],
                occurrence_count: 2,
                aggregate_features: vec![],
                linked_evidence_summaries: vec![],
                representative_snippets: vec!["Snippet".to_string()],
            }],
            structured_fields: vec![LabFieldRecord {
                id: "field-1".to_string(),
                label: "role".to_string(),
                value: "captain".to_string(),
            }],
            definitions: vec![LabDefinitionRecord {
                id: "definition-1".to_string(),
                term: "tau".to_string(),
                definition: "A sector rating".to_string(),
            }],
            section_summary_seeds: vec![LabSummarySeedRecord {
                id: "summary-1".to_string(),
                scope: "section:0".to_string(),
                text: "Summary".to_string(),
            }],
        }
    }
}
