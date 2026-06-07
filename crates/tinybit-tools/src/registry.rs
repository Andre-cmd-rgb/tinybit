use crate::builtin::{
    calc_tool::CalcTool, calendar_tool::CalendarTool, lookup_tool::LookupTool,
    notes_tool::NotesTool, time_tool::TimeTool, todos_tool::TodosTool,
};
use crate::parser::ToolCall;
use crate::tool::{Tool, ToolOutput};
use std::collections::HashMap;

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn execute(&self, call: &ToolCall) -> anyhow::Result<ToolOutput> {
        let tool = self
            .tools
            .get(&call.tool)
            .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", call.tool))?;
        let args = call.args.to_string();
        tool.execute(&args)
    }

    /// Build the tools section of the system prompt.
    pub fn system_prompt_section(&self) -> String {
        let mut out = String::from("Available tools:\n");
        for tool in self.tools.values() {
            out.push_str(&format!(
                "- {}: {} (schema: {})\n",
                tool.name(),
                tool.description(),
                tool.args_schema()
            ));
        }
        out
    }

    /// Register all built-in tools. Creates `data_dir` if missing (the
    /// SQLite-backed tools open db files inside it), so `chat`/`eval` work out
    /// of the box without the user pre-creating the directory.
    pub fn with_builtins(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(data_dir).map_err(|e| {
            anyhow::anyhow!("could not create tools data dir {}: {e}", data_dir.display())
        })?;
        let mut reg = Self::new();
        reg.register(Box::new(TimeTool));
        reg.register(Box::new(CalcTool));
        reg.register(Box::new(LookupTool::new(data_dir)?));
        reg.register(Box::new(TodosTool::new(data_dir)?));
        reg.register(Box::new(NotesTool::new(data_dir)?));
        reg.register(Box::new(CalendarTool::new(data_dir)?));
        Ok(reg)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
