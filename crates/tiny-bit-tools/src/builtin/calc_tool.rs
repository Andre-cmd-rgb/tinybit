use crate::tool::{Tool, ToolOutput};

pub struct CalcTool;

#[derive(serde::Deserialize)]
struct CalcArgs {
    expr: String,
}

impl Tool for CalcTool {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Evaluates a math expression. Supports +,-,*,/,^,sqrt(),sin(),cos(),log(),pi,e" }
    fn args_schema(&self) -> &str { r#"{"expr":"string"}"# }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let parsed: CalcArgs = serde_json::from_str(args)
            .map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        match evalexpr::eval(&parsed.expr) {
            Ok(val) => Ok(ToolOutput::ok(val.to_string())),
            Err(e) => Ok(ToolOutput::err(format!("eval error: {e}"))),
        }
    }
}
