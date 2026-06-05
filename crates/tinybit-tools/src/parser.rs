use crate::tool::ToolOutput;
use std::borrow::Cow;

pub const CALL_START: &str = "<|tool_call|>";
pub const CALL_END: &str = "<|end_tool_call|>";
pub const RESULT_START: &str = "<|tool_result|>";
pub const RESULT_END: &str = "<|end_tool_result|>";

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

/// Format a tool result for injection into context (the format the model was
/// trained on). This is fed back to the model, NOT shown to the user.
pub fn format_tool_result(output: &ToolOutput) -> String {
    format!("{RESULT_START}{}{RESULT_END}", output.content)
}

/// Remove tinybit's tool-protocol markers from text meant for display or saving.
/// The model — trained on the protocol — echoes these back, often malformed
/// (e.g. a half-written `<|end`), and they must never reach the user:
///  - at a tool-call start (`<|tool_call|>`) everything onward is dropped (the
///    call is handled structurally, not shown);
///  - a complete `<|…|>` marker is removed;
///  - a dangling `<|…` fragment is removed up to the next whitespace.
/// Borrows the input unchanged when it contains no `<|` at all.
pub fn strip_tool_markers(text: &str) -> Cow<'_, str> {
    if !text.contains("<|") {
        return Cow::Borrowed(text);
    }
    // Longest real marker is "<|end_tool_result|>" (19 chars); a "|>" further off
    // than this isn't a marker close, so treat the "<|" as a dangling fragment.
    const MAX_MARKER: usize = 24;
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find("<|") {
        out.push_str(&rest[..pos]);
        let frag = &rest[pos..];
        if frag.starts_with(CALL_START) {
            return Cow::Owned(out); // drop the tool call and everything after it
        }
        rest = match frag.find("|>") {
            Some(close) if close + 2 <= MAX_MARKER => &frag[close + 2..],
            _ => match frag[2..].find(char::is_whitespace) {
                Some(ws) => &frag[2 + ws..], // keep the whitespace
                None => "",
            },
        };
    }
    out.push_str(rest);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_is_noop_without_markers() {
        assert!(matches!(strip_tool_markers("hello world"), Cow::Borrowed(_)));
    }

    #[test]
    fn strip_removes_complete_result_markers() {
        assert_eq!(strip_tool_markers("ok<|tool_result|>3<|end_tool_result|>!"), "ok3!");
    }

    #[test]
    fn strip_removes_malformed_dangling_fragments() {
        // The exact leaks seen in chat: a half-written end marker mid-text.
        assert_eq!(strip_tool_markers("#59<|end \"work\""), "#59 \"work\"");
        assert_eq!(strip_tool_markers("n<|ends 96"), "n 96");
        assert_eq!(strip_tool_markers("x<|end"), "x");
    }

    #[test]
    fn strip_drops_tool_call_and_everything_after() {
        assert_eq!(
            strip_tool_markers("before <|tool_call|>{\"tool\":\"x\"}<|end_tool_call|> after"),
            "before "
        );
    }
}
