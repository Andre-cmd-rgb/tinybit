/// Every tool implements this trait.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON schema describing the args field.
    fn args_schema(&self) -> &str;
    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput>;
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: true }
    }
}
