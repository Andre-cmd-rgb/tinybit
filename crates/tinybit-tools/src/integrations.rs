//! The tinybit data API: a language-agnostic, file-based store external apps
//! push user data into (smartwatch metrics, health, app counters — anything).
//!
//! The API **is** the filesystem (decision 19: no server). Layout, per source:
//!
//! ```text
//! data/integrations/<source>/
//!   events.jsonl   append-only event log (rotated when > ~10 MB)
//!   latest.json    {"<metric>": {"ts": ..., "value": ..., "unit": ...}} snapshot
//!   meta.json      {"source": ..., "schema_version": 1, "created": ...}
//! ```
//!
//! One event per line: `{"ts": RFC3339|unix-seconds, "metric": "[a-z0-9_]",
//! "value": number|string, "unit"?: string, "tags"?: object}`. `ts` defaults
//! to now. The recommended writer is `tinybit ingest` (or this module), which
//! validates, appends with O_APPEND semantics, and rewrites `latest.json` via
//! temp-file + atomic rename — but any program in any language may append
//! complete JSON lines directly; readers skip and count malformed lines, never
//! fail on them. The full contract lives in INTEGRATIONS.md.

use chrono::{DateTime, Duration, Utc};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// Rotate `events.jsonl` past this size so a chatty integration can't grow a
/// single file forever. Rotated files stay in the source dir and are still
/// read by range queries.
const ROTATE_BYTES: u64 = 10 * 1024 * 1024;
/// Cap on a string value's length — events are metrics, not documents.
const MAX_VALUE_CHARS: usize = 256;

/// A validated, normalized event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Event {
    pub ts: DateTime<Utc>,
    pub metric: String,
    pub value: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<serde_json::Map<String, serde_json::Value>>,
}

/// What `ingest` did, for the CLI's one-line report.
#[derive(Debug)]
pub struct IngestReport {
    pub source: String,
    pub events: usize,
    pub metrics: usize,
    pub skipped: usize,
}

/// Aggregates over a metric's events in a time range.
#[derive(Debug)]
pub struct RangeSummary {
    pub count: usize,
    /// min/max/mean over NUMERIC values only (string values are counted but
    /// not aggregated).
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub first_ts: Option<DateTime<Utc>>,
    pub last_ts: Option<DateTime<Utc>>,
    /// The most recent few events (≤ 5), oldest → newest.
    pub last: Vec<Event>,
    /// Malformed lines encountered while reading (skipped, never fatal).
    pub skipped_lines: usize,
}

/// Snapshot entry in `latest.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatestEntry {
    pub ts: DateTime<Utc>,
    pub value: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

pub struct IntegrationsStore {
    root: PathBuf,
}

fn valid_source(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 32
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn valid_metric(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// Parse a time spec: RFC3339, unix seconds, or the relative forms a tiny
/// model can reliably emit — `today`, `yesterday`, `now`, or `<N>d`/`<N>h`
/// (that long before `now`).
pub fn parse_time_spec(s: &str, now: DateTime<Utc>) -> anyhow::Result<DateTime<Utc>> {
    let s = s.trim();
    match s.to_ascii_lowercase().as_str() {
        "now" => return Ok(now),
        "today" => {
            return Ok(now
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .map(|d| d.and_utc())
                .unwrap_or(now));
        }
        "yesterday" => {
            return Ok((now - Duration::days(1))
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .map(|d| d.and_utc())
                .unwrap_or(now));
        }
        _ => {}
    }
    if let Some(num) = s
        .strip_suffix('d')
        .and_then(|n| n.parse::<i64>().ok())
        .filter(|&n| (0..=3650).contains(&n))
    {
        return Ok(now - Duration::days(num));
    }
    if let Some(num) = s
        .strip_suffix('h')
        .and_then(|n| n.parse::<i64>().ok())
        .filter(|&n| (0..=87600).contains(&n))
    {
        return Ok(now - Duration::hours(num));
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Bare date → midnight UTC.
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        if let Some(dt) = d.and_hms_opt(0, 0, 0) {
            return Ok(dt.and_utc());
        }
    }
    if let Ok(secs) = s.parse::<i64>() {
        if let Some(dt) = DateTime::from_timestamp(secs, 0) {
            return Ok(dt);
        }
    }
    anyhow::bail!("unrecognized time spec '{s}' (RFC3339, YYYY-MM-DD, unix seconds, today, yesterday, now, <N>d, <N>h)")
}

/// Parse one raw incoming event object (the external-writer schema).
fn parse_raw_event(v: &serde_json::Value, now: DateTime<Utc>) -> anyhow::Result<Event> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("event must be a JSON object"))?;
    let metric = obj
        .get("metric")
        .and_then(|m| m.as_str())
        .ok_or_else(|| anyhow::anyhow!("event missing string field 'metric'"))?
        .to_ascii_lowercase();
    anyhow::ensure!(valid_metric(&metric), "invalid metric name '{metric}' ([a-z0-9_]{{1,64}})");
    let value = obj
        .get("value")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("event missing field 'value'"))?;
    match &value {
        serde_json::Value::Number(_) => {}
        serde_json::Value::String(s) => {
            anyhow::ensure!(s.chars().count() <= MAX_VALUE_CHARS, "string value too long (> {MAX_VALUE_CHARS} chars)");
        }
        serde_json::Value::Bool(_) => {}
        _ => anyhow::bail!("'value' must be a number, string, or bool"),
    }
    let ts = match obj.get("ts") {
        None | Some(serde_json::Value::Null) => now,
        Some(serde_json::Value::String(s)) => parse_time_spec(s, now)?,
        Some(serde_json::Value::Number(n)) => {
            let secs = n.as_i64().ok_or_else(|| anyhow::anyhow!("'ts' number must be unix seconds"))?;
            DateTime::from_timestamp(secs, 0).ok_or_else(|| anyhow::anyhow!("'ts' out of range"))?
        }
        Some(other) => anyhow::bail!("'ts' must be an RFC3339 string or unix seconds, got {other}"),
    };
    let unit = obj.get("unit").and_then(|u| u.as_str()).map(|u| u.to_string());
    let tags = obj.get("tags").and_then(|t| t.as_object()).cloned();
    Ok(Event { ts, metric, value, unit, tags })
}

impl IntegrationsStore {
    /// Open the store under `<data_dir>/integrations`. Nothing is created
    /// until the first ingest.
    pub fn open(data_dir: &Path) -> Self {
        Self { root: data_dir.join("integrations") }
    }

    fn source_dir(&self, source: &str) -> PathBuf {
        self.root.join(source)
    }

    /// Ingest raw JSON for `source`: a single event object, an array of
    /// events, or JSONL (one object per line). Invalid events are skipped and
    /// counted; at least one event must be valid.
    pub fn ingest(&self, source: &str, raw: &str) -> anyhow::Result<IngestReport> {
        anyhow::ensure!(
            valid_source(source),
            "invalid source name '{source}' ([a-z0-9_-]{{1,32}}, lowercase)"
        );
        let now = Utc::now();
        let mut events: Vec<Event> = Vec::new();
        let mut skipped = 0usize;

        let trimmed = raw.trim();
        let parsed: Vec<serde_json::Value> = if trimmed.starts_with('[') {
            serde_json::from_str(trimmed).map_err(|e| anyhow::anyhow!("invalid JSON array: {e}"))?
        } else if trimmed.starts_with('{') && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
        {
            vec![serde_json::from_str(trimmed)?]
        } else {
            // JSONL: parse line by line, skipping (and counting) bad lines.
            trimmed
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| match serde_json::from_str(l) {
                    Ok(v) => Some(v),
                    Err(_) => {
                        skipped += 1;
                        None
                    }
                })
                .collect()
        };
        for v in &parsed {
            match parse_raw_event(v, now) {
                Ok(e) => events.push(e),
                Err(err) => {
                    skipped += 1;
                    eprintln!("ingest: skipping event: {err}");
                }
            }
        }
        anyhow::ensure!(
            !events.is_empty(),
            "no valid events in input ({skipped} skipped)"
        );

        let dir = self.source_dir(source);
        std::fs::create_dir_all(&dir)?;

        // meta.json on first write.
        let meta_path = dir.join("meta.json");
        if !meta_path.exists() {
            let meta = serde_json::json!({
                "source": source,
                "schema_version": 1,
                "created": now.to_rfc3339(),
            });
            write_atomic(&meta_path, &serde_json::to_vec_pretty(&meta)?)?;
        }

        // Rotate a huge log before appending.
        let events_path = dir.join("events.jsonl");
        if let Ok(m) = std::fs::metadata(&events_path) {
            if m.len() > ROTATE_BYTES {
                let rotated = dir.join(format!("events-{}.jsonl", now.timestamp()));
                let _ = std::fs::rename(&events_path, &rotated);
            }
        }

        // Append events (one complete line each — the cross-process contract).
        {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&events_path)?;
            let mut buf = String::new();
            for e in &events {
                buf.push_str(&serde_json::to_string(e)?);
                buf.push('\n');
            }
            f.write_all(buf.as_bytes())?;
        }

        // Update latest.json (read-modify-write, atomic rename).
        let latest_path = dir.join("latest.json");
        let mut latest: BTreeMap<String, LatestEntry> = std::fs::read_to_string(&latest_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        for e in &events {
            let newer = latest.get(&e.metric).map_or(true, |cur| e.ts >= cur.ts);
            if newer {
                latest.insert(
                    e.metric.clone(),
                    LatestEntry { ts: e.ts, value: e.value.clone(), unit: e.unit.clone() },
                );
            }
        }
        write_atomic(&latest_path, &serde_json::to_vec_pretty(&latest)?)?;

        let metrics = events
            .iter()
            .map(|e| e.metric.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        Ok(IngestReport { source: source.to_string(), events: events.len(), metrics, skipped })
    }

    /// All source names (sorted).
    pub fn sources(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        out.push(name.to_string());
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Latest snapshot entries: every (source, metric) pair, optionally
    /// filtered by source and/or metric. Sorted by source then metric.
    pub fn latest(
        &self,
        source: Option<&str>,
        metric: Option<&str>,
    ) -> Vec<(String, String, LatestEntry)> {
        let sources: Vec<String> = match source {
            Some(s) => vec![s.to_string()],
            None => self.sources(),
        };
        let mut out = Vec::new();
        for s in sources {
            let latest_path = self.source_dir(&s).join("latest.json");
            let Some(map) = std::fs::read_to_string(&latest_path)
                .ok()
                .and_then(|t| serde_json::from_str::<BTreeMap<String, LatestEntry>>(&t).ok())
            else {
                continue;
            };
            for (m, entry) in map {
                if metric.map_or(true, |want| m == want) {
                    out.push((s.clone(), m, entry));
                }
            }
        }
        out
    }

    /// Aggregate `metric` events in `[since, until]`, across all sources or
    /// one. Streams the JSONL logs; malformed lines are skipped and counted.
    pub fn range(
        &self,
        source: Option<&str>,
        metric: &str,
        since: DateTime<Utc>,
        until: DateTime<Utc>,
    ) -> RangeSummary {
        let sources: Vec<String> = match source {
            Some(s) => vec![s.to_string()],
            None => self.sources(),
        };
        let mut events: Vec<Event> = Vec::new();
        let mut skipped_lines = 0usize;
        for s in &sources {
            let dir = self.source_dir(s);
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !(name.starts_with("events") && name.ends_with(".jsonl")) {
                    continue;
                }
                let Ok(f) = std::fs::File::open(&path) else { continue };
                for line in std::io::BufReader::new(f).lines() {
                    let Ok(line) = line else {
                        skipped_lines += 1;
                        continue;
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Event>(&line) {
                        Ok(e) => {
                            if e.metric == metric && e.ts >= since && e.ts <= until {
                                events.push(e);
                            }
                        }
                        Err(_) => skipped_lines += 1,
                    }
                }
            }
        }
        events.sort_by_key(|e| e.ts);

        let numeric: Vec<f64> = events.iter().filter_map(|e| e.value.as_f64()).collect();
        let (min, max, mean) = if numeric.is_empty() {
            (None, None, None)
        } else {
            let min = numeric.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = numeric.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = numeric.iter().sum::<f64>() / numeric.len() as f64;
            (Some(min), Some(max), Some(mean))
        };
        let first_ts = events.first().map(|e| e.ts);
        let last_ts = events.last().map(|e| e.ts);
        let tail_start = events.len().saturating_sub(5);
        let last = events[tail_start..].to_vec();
        RangeSummary {
            count: events.len(),
            min,
            max,
            mean,
            first_ts,
            last_ts,
            last,
            skipped_lines,
        }
    }
}

/// Write via temp file + rename so concurrent readers never see a torn file.
fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(name: &str) -> (IntegrationsStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "tinybit-integ-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (IntegrationsStore::open(&dir), dir)
    }

    #[test]
    fn ingest_single_object_and_latest() {
        let (st, _d) = store("single");
        let rep = st
            .ingest("watch", r#"{"metric":"heart_rate","value":61,"unit":"bpm"}"#)
            .unwrap();
        assert_eq!(rep.events, 1);
        assert_eq!(rep.metrics, 1);
        assert_eq!(rep.skipped, 0);
        let latest = st.latest(None, None);
        assert_eq!(latest.len(), 1);
        let (src, metric, entry) = &latest[0];
        assert_eq!(src, "watch");
        assert_eq!(metric, "heart_rate");
        assert_eq!(entry.value, serde_json::json!(61));
        assert_eq!(entry.unit.as_deref(), Some("bpm"));
    }

    #[test]
    fn ingest_array_and_jsonl() {
        let (st, _d) = store("formats");
        let rep = st
            .ingest(
                "watch",
                r#"[{"metric":"steps","value":4102,"ts":"2026-06-08T08:00:00Z"},
                    {"metric":"steps","value":11873,"ts":"2026-06-09T21:00:00Z"}]"#,
            )
            .unwrap();
        assert_eq!(rep.events, 2);
        let rep2 = st
            .ingest(
                "watch",
                "{\"metric\":\"steps\",\"value\":8204,\"ts\":\"2026-06-10T12:00:00Z\"}\nnot json at all\n{\"metric\":\"sleep_hours\",\"value\":7.5,\"ts\":\"2026-06-10T07:00:00Z\"}",
            )
            .unwrap();
        assert_eq!(rep2.events, 2);
        assert_eq!(rep2.skipped, 1);
        // latest.json tracks the newest per metric.
        let latest = st.latest(Some("watch"), Some("steps"));
        assert_eq!(latest.len(), 1);
        assert_eq!(latest[0].2.value, serde_json::json!(8204));
    }

    #[test]
    fn bad_names_and_values_rejected() {
        let (st, _d) = store("invalid");
        assert!(st.ingest("Watch!", r#"{"metric":"x","value":1}"#).is_err()); // bad source
        assert!(st
            .ingest("watch", r#"{"metric":"Heart Rate","value":1}"#)
            .is_err()); // bad metric (space) → 0 valid events
        assert!(st.ingest("watch", r#"{"metric":"hr"}"#).is_err()); // missing value
        assert!(st.ingest("watch", r#"{"metric":"hr","value":{"nested":1}}"#).is_err());
        // Metric names are normalized to lowercase.
        let rep = st.ingest("watch", r#"{"metric":"HEART_RATE","value":60}"#).unwrap();
        assert_eq!(rep.events, 1);
        assert_eq!(st.latest(None, Some("heart_rate")).len(), 1);
    }

    #[test]
    fn range_aggregates_and_relative_times() {
        let (st, _d) = store("range");
        let now = Utc::now();
        let mk = |hours_ago: i64, v: i64| {
            format!(
                r#"{{"metric":"steps","value":{v},"ts":"{}"}}"#,
                (now - Duration::hours(hours_ago)).to_rfc3339()
            )
        };
        let lines = [mk(30, 4102), mk(20, 11873), mk(2, 8204)].join("\n");
        st.ingest("watch", &lines).unwrap();

        let since = parse_time_spec("7d", now).unwrap();
        let sum = st.range(None, "steps", since, now);
        assert_eq!(sum.count, 3);
        assert_eq!(sum.min, Some(4102.0));
        assert_eq!(sum.max, Some(11873.0));
        assert!((sum.mean.unwrap() - 8059.666).abs() < 0.01);
        assert_eq!(sum.last.len(), 3);
        assert_eq!(sum.skipped_lines, 0);

        // A tighter window excludes the oldest event (30h ago).
        let since_1d = parse_time_spec("24h", now).unwrap();
        let sum = st.range(Some("watch"), "steps", since_1d, now);
        assert_eq!(sum.count, 2);
        assert_eq!(sum.min, Some(8204.0));
        assert_eq!(sum.max, Some(11873.0));
    }

    #[test]
    fn malformed_log_lines_are_skipped_not_fatal() {
        let (st, dir) = store("malformed");
        st.ingest("watch", r#"{"metric":"hr","value":60}"#).unwrap();
        // A foreign writer appends garbage directly.
        let log = dir.join("integrations/watch/events.jsonl");
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        writeln!(f, "{{ totally broken").unwrap();
        writeln!(f, r#"{{"metric":"hr","value":62,"ts":"{}"}}"#, Utc::now().to_rfc3339()).unwrap();
        let sum = st.range(None, "hr", Utc::now() - Duration::days(1), Utc::now());
        assert_eq!(sum.count, 2);
        assert_eq!(sum.skipped_lines, 1);
    }

    #[test]
    fn time_spec_forms() {
        let now = DateTime::parse_from_rfc3339("2026-06-10T15:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(parse_time_spec("now", now).unwrap(), now);
        assert_eq!(
            parse_time_spec("today", now).unwrap().to_rfc3339(),
            "2026-06-10T00:00:00+00:00"
        );
        assert_eq!(
            parse_time_spec("yesterday", now).unwrap().to_rfc3339(),
            "2026-06-09T00:00:00+00:00"
        );
        assert_eq!(parse_time_spec("7d", now).unwrap(), now - Duration::days(7));
        assert_eq!(parse_time_spec("12h", now).unwrap(), now - Duration::hours(12));
        assert_eq!(
            parse_time_spec("2026-06-01", now).unwrap().to_rfc3339(),
            "2026-06-01T00:00:00+00:00"
        );
        assert!(parse_time_spec("2026-06-01T10:00:00Z", now).is_ok());
        assert!(parse_time_spec("not a time", now).is_err());
    }

    #[test]
    fn latest_json_written_atomically() {
        let (st, dir) = store("atomic");
        st.ingest("scale", r#"{"metric":"weight","value":72.4,"unit":"kg"}"#).unwrap();
        let path = dir.join("integrations/scale/latest.json");
        assert!(path.exists());
        assert!(!path.with_extension("json.tmp").exists(), "temp file left behind");
        let parsed: BTreeMap<String, LatestEntry> =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(parsed.contains_key("weight"));
    }
}
