use crate::tool::{Tool, ToolOutput};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Bundled default knowledge base, compiled into the binary. Users extend it by
/// dropping a `knowledge.json` with the same shape into the tools data dir.
const BUNDLED_KNOWLEDGE: &str = include_str!("../../data/knowledge.json");

#[derive(serde::Deserialize)]
struct RawEntry {
    q: String,
    a: String,
    #[serde(default)]
    alt: Vec<String>,
}

#[derive(serde::Deserialize)]
struct LookupArgs {
    query: String,
}

/// One answer plus the phrases (canonical question + aliases) that should match
/// it, each pre-tokenized to its significant tokens.
struct Entry {
    answer: String,
    phrases: Vec<Vec<String>>,
}

/// Local read-only fact lookup. A tiny model can't *store* facts reliably, but
/// it can learn to *fetch* them: this returns the best-matching curated fact for
/// a query, or a clear "not found" so the model doesn't bluff.
pub struct LookupTool {
    entries: Vec<Entry>,
    idf:     HashMap<String, f64>,
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "whats", "who", "whom", "when", "where",
    "why", "how", "which", "many", "much", "does", "do", "did", "tell", "me",
    "please", "it", "its", "there", "that", "this", "with", "by", "about", "as",
    "from", "your", "you", "yourself", "we", "they", "he", "she", "my",
];

fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| w.len() > 1 && !STOPWORDS.contains(w))
        .map(str::to_string)
        .collect()
}

/// Two tokens match if equal, or one is a ≥4-char prefix of the other — cheap
/// stemming so "tall"~"tallest" and "mount"~"mountain".
fn tok_match(a: &str, b: &str) -> bool {
    a == b || (a.len() >= 4 && b.starts_with(a)) || (b.len() >= 4 && a.starts_with(b))
}

impl LookupTool {
    pub fn new(data_dir: &Path) -> anyhow::Result<Self> {
        let mut raw: Vec<RawEntry> = serde_json::from_str(BUNDLED_KNOWLEDGE)
            .map_err(|e| anyhow::anyhow!("bundled knowledge.json is invalid: {e}"))?;

        // Optional user extension: data_dir/knowledge.json (same shape).
        let user = data_dir.join("knowledge.json");
        if user.exists() {
            match std::fs::read_to_string(&user)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str::<Vec<RawEntry>>(&t).map_err(|e| e.to_string()))
            {
                Ok(mut extra) => raw.append(&mut extra),
                Err(e) => eprintln!("lookup: ignoring {}: {e}", user.display()),
            }
        }

        let mut entries = Vec::with_capacity(raw.len());
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut n_phrases = 0usize;
        for r in raw {
            let mut phrases = Vec::new();
            for p in std::iter::once(r.q.clone()).chain(r.alt.iter().cloned()) {
                let toks = tokenize(&p);
                if toks.is_empty() {
                    continue;
                }
                n_phrases += 1;
                for t in toks.iter().collect::<HashSet<_>>() {
                    *df.entry(t.clone()).or_insert(0) += 1;
                }
                phrases.push(toks);
            }
            if !phrases.is_empty() {
                entries.push(Entry { answer: r.a, phrases });
            }
        }
        let n = n_phrases.max(1) as f64;
        let idf = df
            .into_iter()
            .map(|(t, c)| (t, (n / c as f64).ln() + 1.0))
            .collect();
        Ok(Self { entries, idf })
    }

    fn idf(&self, t: &str) -> f64 {
        self.idf.get(t).copied().unwrap_or(3.0)
    }

    /// Best answer whose match clears the IDF-coverage threshold, if any. IDF
    /// weighting is what stops a shared generic token ("capital") from matching
    /// the wrong specific entry — only sharing the distinctive token (the
    /// country) clears the bar.
    fn best_match(&self, query: &str) -> Option<&str> {
        let q = tokenize(query);
        if q.is_empty() {
            return None;
        }
        let mut best: Option<(f64, &str)> = None;
        for e in &self.entries {
            for phrase in &e.phrases {
                let total: f64 = phrase.iter().map(|t| self.idf(t)).sum();
                if total <= 0.0 {
                    continue;
                }
                let matched: f64 = phrase
                    .iter()
                    .filter(|pt| q.iter().any(|qt| tok_match(qt, pt)))
                    .map(|pt| self.idf(pt))
                    .sum();
                let ratio = matched / total;
                if ratio >= 0.6 {
                    let score = matched + ratio; // weight first, coverage as tie-break
                    if best.map_or(true, |(b, _)| score > b) {
                        best = Some((score, e.answer.as_str()));
                    }
                }
            }
        }
        best.map(|(_, a)| a)
    }
}

impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "Look up a fact from the local knowledge base (capitals, geography, science, space, definitions). Use it for factual questions instead of guessing."
    }
    fn args_schema(&self) -> &str {
        r#"{"query":"string"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let parsed: LookupArgs =
            serde_json::from_str(args).map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        match self.best_match(&parsed.query) {
            Some(answer) => Ok(ToolOutput::ok(answer.to_string())),
            None => Ok(ToolOutput::ok(format!(
                "No local entry for \"{}\".",
                parsed.query.trim()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> LookupTool {
        // data_dir without a knowledge.json → bundled set only.
        LookupTool::new(Path::new("/nonexistent-tinybit-data")).unwrap()
    }

    fn ask(t: &LookupTool, q: &str) -> String {
        let args = serde_json::json!({ "query": q }).to_string();
        t.execute(&args).unwrap().content
    }

    #[test]
    fn finds_capital() {
        assert!(ask(&tool(), "what is the capital of italy?").contains("Rome"));
        assert!(ask(&tool(), "capital of japan").contains("Tokyo"));
    }

    #[test]
    fn distinctive_token_prevents_wrong_capital() {
        // "capital" is shared by ~33 entries; only matching the country should win.
        let a = ask(&tool(), "what's the capital of france");
        assert!(a.contains("Paris"), "got: {a}");
        assert!(!a.contains("Rome"), "matched the wrong capital: {a}");
    }

    #[test]
    fn fuzzy_prefix_match() {
        // "tall"~"tallest", "mount"~"mountain"
        assert!(ask(&tool(), "how tall is mount everest").contains("Everest"));
    }

    #[test]
    fn known_science_fact() {
        assert!(ask(&tool(), "chemical formula for water").contains("H2O"));
        assert!(ask(&tool(), "how many continents are there").contains("seven"));
    }

    #[test]
    fn unknown_returns_not_found_not_a_bluff() {
        let a = ask(&tool(), "what is the airspeed velocity of an unladen swallow");
        assert!(a.starts_with("No local entry"), "got: {a}");
    }

    #[test]
    fn identity_query_does_not_false_match() {
        // "who are you" tokenizes to nothing significant → no spurious fact.
        assert!(ask(&tool(), "who are you").starts_with("No local entry"));
    }
}
