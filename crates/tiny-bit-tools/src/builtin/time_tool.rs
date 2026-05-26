use crate::tool::{Tool, ToolOutput};
use chrono::Local;

pub struct TimeTool;

impl Tool for TimeTool {
    fn name(&self) -> &str { "time" }
    fn description(&self) -> &str { "Returns current date, time, day of week, and timezone" }
    fn args_schema(&self) -> &str { "{}" }

    fn execute(&self, _args: &str) -> anyhow::Result<ToolOutput> {
        let now = Local::now();
        let formatted = now.format("%Y-%m-%d %H:%M:%S %A %Z").to_string();
        Ok(ToolOutput::ok(formatted))
    }
}
