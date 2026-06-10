//! `user_data` — the model's read side of the integrations data API
//! (see `crate::integrations` and INTEGRATIONS.md). External apps push events
//! (smartwatch, scale, anything) into `data/integrations/`; this tool lets the
//! model FETCH them and reason over what it gets — never invent numbers.

use crate::integrations::{parse_time_spec, IntegrationsStore, LatestEntry};
use crate::tool::{Tool, ToolOutput};
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use std::path::Path;

pub struct UserDataTool {
    store: IntegrationsStore,
}

#[derive(serde::Deserialize)]
struct Args {
    action: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    metric: Option<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default)]
    until: Option<String>,
}

/// Compact number formatting for one-line tool results: integers stay
/// integers, fractions get one decimal.
fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.1}")
    }
}

fn fmt_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Number(n) => n
            .as_f64()
            .map(fmt_num)
            .unwrap_or_else(|| n.to_string()),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn fmt_ts(ts: &DateTime<Utc>) -> String {
    ts.format("%Y-%m-%dT%H:%MZ").to_string()
}

fn fmt_date(ts: &DateTime<Utc>) -> String {
    ts.format("%Y-%m-%d").to_string()
}

impl UserDataTool {
    pub fn new(data_dir: &Path) -> Self {
        Self { store: IntegrationsStore::open(data_dir) }
    }

    /// `latest`: one line per (source, metric):
    /// `heart_rate: 61 bpm at 2026-06-10T08:31Z (watch)`
    fn latest(&self, source: Option<&str>, metric: Option<&str>) -> ToolOutput {
        let entries = self.store.latest(source, metric);
        if entries.is_empty() {
            let what = metric.or(source).unwrap_or("user data");
            return ToolOutput::ok(format!("No data for \"{what}\"."));
        }
        let lines: Vec<String> = entries
            .iter()
            .map(|(src, m, LatestEntry { ts, value, unit })| {
                let unit = unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
                format!("{m}: {}{unit} at {} ({src})", fmt_value(value), fmt_ts(ts))
            })
            .collect();
        ToolOutput::ok(lines.join("\n"))
    }

    /// `range`: one summary line:
    /// `steps 2026-06-03→2026-06-10: count=14 min=4102 max=11873 mean=8204 last=9120`
    fn range(
        &self,
        source: Option<&str>,
        metric: &str,
        since: &str,
        until: &str,
    ) -> anyhow::Result<ToolOutput> {
        let now = Utc::now();
        let since_dt = parse_time_spec(since, now)?;
        let until_dt = parse_time_spec(until, now)?;
        let sum = self.store.range(source, metric, since_dt, until_dt);
        if sum.count == 0 {
            return Ok(ToolOutput::ok(format!("No data for \"{metric}\".")));
        }
        let mut line = format!(
            "{metric} {}→{}: count={}",
            fmt_date(&since_dt),
            fmt_date(&until_dt),
            sum.count
        );
        if let (Some(min), Some(max), Some(mean)) = (sum.min, sum.max, sum.mean) {
            line.push_str(&format!(
                " min={} max={} mean={}",
                fmt_num(min),
                fmt_num(max),
                fmt_num(mean)
            ));
        }
        if let Some(last) = sum.last.last() {
            line.push_str(&format!(" last={}", fmt_value(&last.value)));
        }
        Ok(ToolOutput::ok(line))
    }

    /// `sources`: `Sources: watch (heart_rate, steps), scale (weight)`
    fn sources(&self) -> ToolOutput {
        let sources = self.store.sources();
        if sources.is_empty() {
            return ToolOutput::ok(
                "No data sources connected yet. Apps can add data with `tinybit ingest`.".to_string(),
            );
        }
        let mut parts = Vec::new();
        for s in sources {
            let metrics: BTreeMap<String, ()> = self
                .store
                .latest(Some(&s), None)
                .into_iter()
                .map(|(_, m, _)| (m, ()))
                .collect();
            let list: Vec<&str> = metrics.keys().map(String::as_str).collect();
            parts.push(format!("{s} ({})", list.join(", ")));
        }
        ToolOutput::ok(format!("Sources: {}", parts.join(", ")))
    }
}

impl Tool for UserDataTool {
    fn name(&self) -> &str {
        "user_data"
    }
    fn description(&self) -> &str {
        "Query the user's connected data (smartwatch, health, app metrics): latest values, time-range stats, or the list of sources. Use it instead of guessing the user's numbers."
    }
    fn args_schema(&self) -> &str {
        r#"{"action":"latest|range|sources","source":"string?","metric":"string?","since":"string?","until":"string?"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let parsed: Args =
            serde_json::from_str(args).map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        let source = parsed.source.as_deref().filter(|s| !s.is_empty());
        let metric = parsed.metric.as_deref().filter(|m| !m.is_empty());
        match parsed.action.as_str() {
            "latest" => Ok(self.latest(source, metric)),
            "range" => {
                let Some(metric) = metric else {
                    return Ok(ToolOutput::err("range needs a \"metric\".".to_string()));
                };
                let since = parsed.since.as_deref().unwrap_or("7d");
                let until = parsed.until.as_deref().unwrap_or("now");
                self.range(source, metric, since, until)
            }
            "sources" => Ok(self.sources()),
            other => Ok(ToolOutput::err(format!(
                "unknown action \"{other}\" (expected latest, range, or sources)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::IntegrationsStore;

    fn seeded(name: &str) -> UserDataTool {
        let dir = std::env::temp_dir().join(format!(
            "tinybit-userdata-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = IntegrationsStore::open(&dir);
        let now = Utc::now();
        let mk = |hours_ago: i64, metric: &str, v: f64, unit: &str| {
            format!(
                r#"{{"metric":"{metric}","value":{v},"unit":"{unit}","ts":"{}"}}"#,
                (now - chrono::Duration::hours(hours_ago)).to_rfc3339()
            )
        };
        store
            .ingest(
                "watch",
                &[
                    mk(30, "steps", 4102.0, "steps"),
                    mk(20, "steps", 11873.0, "steps"),
                    mk(2, "steps", 8204.0, "steps"),
                    mk(1, "heart_rate", 61.0, "bpm"),
                ]
                .join("\n"),
            )
            .unwrap();
        store
            .ingest("scale", r#"{"metric":"weight","value":72.4,"unit":"kg"}"#)
            .unwrap();
        UserDataTool::new(&dir)
    }

    fn run(t: &UserDataTool, args: serde_json::Value) -> ToolOutput {
        t.execute(&args.to_string()).unwrap()
    }

    #[test]
    fn latest_lists_metrics_with_units_and_sources() {
        let t = seeded("latest");
        let out = run(&t, serde_json::json!({"action":"latest"}));
        assert!(!out.is_error);
        assert!(out.content.contains("heart_rate: 61 bpm at"), "got: {}", out.content);
        assert!(out.content.contains("(watch)"));
        assert!(out.content.contains("weight: 72.4 kg at"));
        assert!(out.content.contains("(scale)"));
        // Filtered by metric.
        let out = run(&t, serde_json::json!({"action":"latest","metric":"heart_rate"}));
        assert!(out.content.contains("heart_rate"));
        assert!(!out.content.contains("weight"));
    }

    #[test]
    fn range_summarizes() {
        let t = seeded("range");
        let out = run(
            &t,
            serde_json::json!({"action":"range","metric":"steps","since":"7d"}),
        );
        assert!(!out.is_error);
        assert!(out.content.contains("count=3"), "got: {}", out.content);
        assert!(out.content.contains("min=4102"));
        assert!(out.content.contains("max=11873"));
        assert!(out.content.contains("mean=8059.7"));
        assert!(out.content.contains("last=8204"));
    }

    #[test]
    fn no_data_is_an_honest_miss() {
        let t = seeded("miss");
        let out = run(&t, serde_json::json!({"action":"latest","metric":"blood_pressure"}));
        assert_eq!(out.content, "No data for \"blood_pressure\".");
        let out = run(&t, serde_json::json!({"action":"range","metric":"blood_pressure"}));
        assert_eq!(out.content, "No data for \"blood_pressure\".");
    }

    #[test]
    fn sources_lists_metrics() {
        let t = seeded("sources");
        let out = run(&t, serde_json::json!({"action":"sources"}));
        assert!(out.content.starts_with("Sources:"), "got: {}", out.content);
        assert!(out.content.contains("watch (heart_rate, steps)"));
        assert!(out.content.contains("scale (weight)"));
    }

    #[test]
    fn empty_store_and_bad_action() {
        let dir = std::env::temp_dir().join(format!("tinybit-userdata-{}-empty", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let t = UserDataTool::new(&dir);
        let out = run(&t, serde_json::json!({"action":"sources"}));
        assert!(out.content.starts_with("No data sources connected"));
        let out = run(&t, serde_json::json!({"action":"latest"}));
        assert!(out.content.starts_with("No data for"));
        let out = run(&t, serde_json::json!({"action":"explode"}));
        assert!(out.is_error);
        let out = run(&t, serde_json::json!({"action":"range"}));
        assert!(out.is_error); // range without metric
    }
}
