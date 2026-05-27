use tinybit_tools::{
    builtin::{
        calc_tool::CalcTool, time_tool::TimeTool, todos_tool::TodosTool,
    },
    parse_tool_call,
    tool::Tool,
};

#[test]
fn test_calc_tool_basic() {
    let tool = CalcTool;
    let result = tool.execute(r#"{"expr":"2+2*3"}"#).unwrap();
    assert_eq!(result.content.trim(), "8");
    assert!(!result.is_error);
}

#[test]
fn test_calc_tool_power() {
    let tool = CalcTool;
    let result = tool.execute(r#"{"expr":"12^2"}"#).unwrap();
    assert!(!result.is_error, "eval failed: {}", result.content);
    let val: f64 = result.content.trim().parse()
        .unwrap_or_else(|_| panic!("not a number: {:?}", result.content));
    assert!((val - 144.0).abs() < 1e-6, "12^2 = {}", val);
}

#[test]
fn test_calc_tool_invalid() {
    let tool = CalcTool;
    let result = tool.execute(r#"{"expr":"1/0"}"#).unwrap();
    // evalexpr returns infinity or error for division by zero
    // Just check it doesn't panic
    let _ = result.content;
}

#[test]
fn test_parse_tool_call() {
    let text = r#"Let me calculate that. <|tool_call|>{"tool":"calculator","args":{"expr":"sqrt(144)"}}<|end_tool_call|>"#;
    let parsed = parse_tool_call(text).unwrap();
    assert_eq!(parsed.0.tool, "calculator");
    assert_eq!(parsed.0.args["expr"], "sqrt(144)");
    assert_eq!(parsed.1, "Let me calculate that. ");
}

#[test]
fn test_parse_tool_call_missing_end() {
    let text = r#"<|tool_call|>{"tool":"time","args":{}}"#;
    assert!(parse_tool_call(text).is_none(), "should return None when end marker missing");
}

#[test]
fn test_todos_add_and_list() {
    let dir = tempdir();
    let tool = TodosTool::new(&dir).unwrap();
    let add1 = tool.execute(r#"{"action":"add","text":"Buy milk"}"#).unwrap();
    assert!(!add1.is_error);
    let add2 = tool.execute(r#"{"action":"add","text":"Write tests"}"#).unwrap();
    assert!(!add2.is_error);

    let list = tool.execute(r#"{"action":"list"}"#).unwrap();
    assert!(list.content.contains("Buy milk"));
    assert!(list.content.contains("Write tests"));

    // Complete first
    let done = tool.execute(r#"{"action":"complete","id":1}"#).unwrap();
    assert!(!done.is_error);

    // List should show done mark
    let list2 = tool.execute(r#"{"action":"list"}"#).unwrap();
    assert!(list2.content.contains("✓"));
}

#[test]
fn test_time_tool_returns_date() {
    let tool = TimeTool;
    let result = tool.execute("{}").unwrap();
    assert!(!result.content.is_empty());
    assert!(result.content.contains("202"), "should contain year 202x: {}", result.content);
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tinybit-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
