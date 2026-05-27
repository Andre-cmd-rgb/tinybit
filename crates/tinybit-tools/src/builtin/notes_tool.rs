use crate::tool::{Tool, ToolOutput};
use rusqlite::{Connection, params};
use std::path::PathBuf;

pub struct NotesTool {
    db_path: PathBuf,
}

impl NotesTool {
    pub fn new(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let db_path = data_dir.join("notes.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
                USING fts5(title, content, content='notes', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content);
            END;"
        )?;
        Ok(Self { db_path })
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "action")]
enum NotesArgs {
    #[serde(rename = "save")]   Save { title: String, content: String },
    #[serde(rename = "search")] Search { query: String },
    #[serde(rename = "get")]    Get { id: i64 },
    #[serde(rename = "list")]   List,
}

impl Tool for NotesTool {
    fn name(&self) -> &str { "notes" }
    fn description(&self) -> &str { "Save and search notes with full-text search" }
    fn args_schema(&self) -> &str {
        r#"{"action":"save|search|get|list","title":"string","content":"string","query":"string","id":"int"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let conn = Connection::open(&self.db_path)?;
        let action: NotesArgs = serde_json::from_str(args)
            .map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        match action {
            NotesArgs::Save { title, content } => {
                conn.execute(
                    "INSERT INTO notes (title, content) VALUES (?1, ?2)",
                    params![title, content],
                )?;
                let id = conn.last_insert_rowid();
                Ok(ToolOutput::ok(format!("Saved note #{id}: {title}")))
            }
            NotesArgs::Search { query } => {
                let mut stmt = conn.prepare(
                    "SELECT n.id, n.title FROM notes n
                     JOIN notes_fts ON notes_fts.rowid = n.id
                     WHERE notes_fts MATCH ?1 LIMIT 10"
                )?;
                let rows: Vec<String> = stmt
                    .query_map(params![query], |row| {
                        let id: i64 = row.get(0)?;
                        let title: String = row.get(1)?;
                        Ok(format!("#{id} {title}"))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok("No matching notes."))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
            NotesArgs::Get { id } => {
                let result: Option<(String, String)> = conn
                    .query_row(
                        "SELECT title, content FROM notes WHERE id=?1",
                        params![id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .ok();
                match result {
                    Some((title, content)) => Ok(ToolOutput::ok(format!("# {title}\n\n{content}"))),
                    None => Ok(ToolOutput::err(format!("No note with id {id}"))),
                }
            }
            NotesArgs::List => {
                let mut stmt = conn.prepare(
                    "SELECT id, title, created_at FROM notes ORDER BY id DESC LIMIT 20"
                )?;
                let rows: Vec<String> = stmt
                    .query_map([], |row| {
                        let id: i64 = row.get(0)?;
                        let title: String = row.get(1)?;
                        let ts: String = row.get(2)?;
                        Ok(format!("#{id} [{ts}] {title}"))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok("No notes."))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
        }
    }
}
