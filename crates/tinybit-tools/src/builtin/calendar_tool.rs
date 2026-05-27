use crate::tool::{Tool, ToolOutput};
use rusqlite::{Connection, params};
use std::path::PathBuf;

pub struct CalendarTool {
    db_path: PathBuf,
}

impl CalendarTool {
    pub fn new(data_dir: &std::path::Path) -> anyhow::Result<Self> {
        let db_path = data_dir.join("calendar.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_date ON events(date);"
        )?;
        Ok(Self { db_path })
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "action")]
enum CalendarArgs {
    #[serde(rename = "add")]    Add { title: String, date: String, #[serde(default)] time: Option<String>, #[serde(default)] notes: Option<String> },
    #[serde(rename = "today")]  Today,
    #[serde(rename = "week")]   Week,
    #[serde(rename = "list")]   List { from: String, to: String },
    #[serde(rename = "delete")] Delete { id: i64 },
}

fn format_event(id: i64, title: &str, date: &str, time: Option<&str>) -> String {
    match time {
        Some(t) => format!("#{id} {date} {t} — {title}"),
        None    => format!("#{id} {date} — {title}"),
    }
}

impl Tool for CalendarTool {
    fn name(&self) -> &str { "calendar" }
    fn description(&self) -> &str { "Manage calendar events (add/today/week/list/delete)" }
    fn args_schema(&self) -> &str {
        r#"{"action":"add|today|week|list|delete","title":"string","date":"YYYY-MM-DD","time":"HH:MM","notes":"string","from":"YYYY-MM-DD","to":"YYYY-MM-DD","id":"int"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let conn = Connection::open(&self.db_path)?;
        let action: CalendarArgs = serde_json::from_str(args)
            .map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        match action {
            CalendarArgs::Add { title, date, time, notes } => {
                conn.execute(
                    "INSERT INTO events (title, date, time, notes) VALUES (?1, ?2, ?3, ?4)",
                    params![title, date, time, notes],
                )?;
                let id = conn.last_insert_rowid();
                Ok(ToolOutput::ok(format!("Added event #{id}: {title} on {date}")))
            }
            CalendarArgs::Today => {
                let today = chrono::Local::now().format("%Y-%m-%d").to_string();
                let mut stmt = conn.prepare(
                    "SELECT id, title, date, time FROM events WHERE date=?1 ORDER BY time"
                )?;
                let rows: Vec<String> = stmt
                    .query_map(params![today], |row| {
                        let id: i64 = row.get(0)?;
                        let title: String = row.get(1)?;
                        let date: String = row.get(2)?;
                        let time: Option<String> = row.get(3)?;
                        Ok(format_event(id, &title, &date, time.as_deref()))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok(format!("No events today ({today}).")))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
            CalendarArgs::Week => {
                let today = chrono::Local::now();
                let from = today.format("%Y-%m-%d").to_string();
                let to = (today + chrono::Duration::days(7)).format("%Y-%m-%d").to_string();
                let mut stmt = conn.prepare(
                    "SELECT id, title, date, time FROM events WHERE date>=?1 AND date<=?2 ORDER BY date, time"
                )?;
                let rows: Vec<String> = stmt
                    .query_map(params![from, to], |row| {
                        let id: i64 = row.get(0)?;
                        let title: String = row.get(1)?;
                        let date: String = row.get(2)?;
                        let time: Option<String> = row.get(3)?;
                        Ok(format_event(id, &title, &date, time.as_deref()))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok("No events this week."))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
            CalendarArgs::List { from, to } => {
                let mut stmt = conn.prepare(
                    "SELECT id, title, date, time FROM events WHERE date>=?1 AND date<=?2 ORDER BY date, time"
                )?;
                let rows: Vec<String> = stmt
                    .query_map(params![from, to], |row| {
                        let id: i64 = row.get(0)?;
                        let title: String = row.get(1)?;
                        let date: String = row.get(2)?;
                        let time: Option<String> = row.get(3)?;
                        Ok(format_event(id, &title, &date, time.as_deref()))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                if rows.is_empty() {
                    Ok(ToolOutput::ok(format!("No events from {from} to {to}.")))
                } else {
                    Ok(ToolOutput::ok(rows.join("\n")))
                }
            }
            CalendarArgs::Delete { id } => {
                let n = conn.execute("DELETE FROM events WHERE id=?1", params![id])?;
                if n == 0 {
                    Ok(ToolOutput::err(format!("No event with id {id}")))
                } else {
                    Ok(ToolOutput::ok(format!("Deleted event #{id}")))
                }
            }
        }
    }
}
