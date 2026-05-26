use crate::tool::ToolOutput;

const CALL_START: &str = "<|tool_call|>";
const CALL_END: &str = "<|end_tool_call|>";

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: serde_json::Value,
}

/// Scan a string for a complete tool call.
/// Returns Some((call, before_text, after_marker)) if found.
pub fn parse_tool_call(text: &str) -> Option<(ToolCall, &str, &str)> {
    let start = text.find(CALL_START)?;
    let content_start = start + CALL_START.len();
    let end = text[content_start..].find(CALL_END)?;
    let json = &text[content_start..content_start + end];
    let after = &text[content_start + end + CALL_END.len()..];
    let before = &text[..start];
    let call: ToolCall = serde_json::from_str(json).ok()?;
    Some((call, before, after))
}

/// Format a tool result for injection into context.
pub fn format_tool_result(output: &ToolOutput) -> String {
    format!("<|tool_result|>{}<|end_tool_result|>", output.content)
}
