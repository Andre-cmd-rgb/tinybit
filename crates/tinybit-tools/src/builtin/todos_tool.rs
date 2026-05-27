use crate::tool::{Tool, ToolOutput};
use rusqlite::{Connection, params};
use std::path::PathBuf;

pub struct TodosTool {
    db_path: PathBuf,
}

impl TodosTool {
    pub fn new(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let db_path = data_dir.join("todos.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        )?;
        Ok(Self { db_path })
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "action")]
enum TodosArgs {
    #[serde(rename = "add")]   Add { text: String },
    #[serde(rename = "list")]  List,
    #[serde(rename = "complete")] Complete { id: i64 },
    #[serde(rename = "delete")]  Delete { id: i64 },
}

impl Tool for TodosTool {
    fn name(&self) -> &str { "todos" }
    fn description(&self) -> &str { "Manage a todo list (add/list/complete/delete)" }
    fn args_schema(&self) -> &str {
        r#"{"action":"add|list|complete|delete","text":"string (add only)","id":"int (complete/delete only)"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let conn = Connection::open(&self.db_path)?;
        let action: TodosArgs = serde_json::from_str(args)
            .map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        match action {
            TodosArgs::Add { text } => {
                conn.execute("INSERT INTO todos (text) VALUES (?1)", params![text])?;
                let id = conn.last_insert_rowid();
                Ok(ToolOutput::ok(format!("Added todo #{id}: {text}")))
            }
            TodosArgs::List => {
                let mut stmt = conn.prepare(
                    "SELECT id, text, done FROM todos ORDER BY id"
                )?;
                let rows: Vec<String> = stmt
                    .query_map([], |row| {
                        let id: i64 = row.get(0)?;
                        let text: String = row.get(1)?;
                        let done: i64 = row.get(2)?;
                        let mark = if done != 0 { "✓" } else { "○" };
                        Ok(format!("{mark} #{id} {text}"))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok("No todos."))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
            TodosArgs::Complete { id } => {
                let n = conn.execute("UPDATE todos SET done=1 WHERE id=?1", params![id])?;
                if n == 0 {
                    Ok(ToolOutput::err(format!("No todo with id {id}")))
                } else {
                    Ok(ToolOutput::ok(format!("Marked #{id} as done")))
                }
            }
            TodosArgs::Delete { id } => {
                let n = conn.execute("DELETE FROM todos WHERE id=?1", params![id])?;
                if n == 0 {
                    Ok(ToolOutput::err(format!("No todo with id {id}")))
                } else {
                    Ok(ToolOutput::ok(format!("Deleted #{id}")))
                }
            }
        }
    }
}
