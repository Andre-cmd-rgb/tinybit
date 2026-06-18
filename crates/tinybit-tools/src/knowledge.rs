//! Local knowledge store — the "hippocampus" of tinybit.
//!
//! A tiny model can't reliably *store* world facts in its weights, but it can
//! learn to *fetch* them. This store holds curated knowledge on disk and
//! retrieves the best-matching passages for a query with IDF-weighted token
//! matching. It backs both the single-answer `lookup` tool and (later) an
//! automatic retrieval-augmented step: search locally, then let the model
//! summarize from what it found instead of memorizing or bluffing.
//!
//! Two entry shapes are supported in `knowledge.json` (mixed freely):
//!   * Q/A facts:    `{"q": "capital of italy", "a": "Rome ...", "alt": [...]}`
//!   * Definitions:  `{"title": "Photosynthesis", "text": "Photosynthesis is ..."}`
//!
//! Q/A entries are matched on *phrase coverage* (how much of the short canonical
//! phrase the query covers — this is what stops a generic shared token like
//! "capital" from matching the wrong country). Definition/document entries are
//! matched on *query coverage* (how much of the query the passage explains),
//! which is the right criterion for longer free text.

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Bundled default knowledge base, compiled into the binary. Users extend it by
/// dropping a `knowledge.json` with the same shape into the tools data dir.
const BUNDLED_KNOWLEDGE: &str = include_str!("../data/knowledge.json");

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "at",
    "to", "for", "and", "or", "what", "whats", "who", "whom", "when", "where",
    "why", "how", "which", "many", "much", "does", "do", "did", "tell", "me",
    "please", "it", "its", "there", "that", "this", "with", "by", "about", "as",
    "from", "your", "you", "yourself", "we", "they", "he", "she", "my",
];

/// Default minimum query/phrase coverage for a passage to be considered a hit.
/// The single-answer `lookup` tool uses a stricter value for precision.
pub const DEFAULT_MIN_COVERAGE: f64 = 0.4;
/// Coverage the legacy single-answer lookup requires (preserves prior behavior).
pub const STRICT_MIN_COVERAGE: f64 = 0.6;

#[derive(serde::Deserialize)]
struct RawEntry {
    #[serde(default)]
    q: Option<String>,
    #[serde(default)]
    a: Option<String>,
    #[serde(default)]
    alt: Vec<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MatchMode {
    /// Short canonical phrases (Q/A): gate on how much of the *phrase* the query covers.
    Phrase,
    /// Free text (definitions/docs): gate on how much of the *query* the passage covers.
    Query,
}

/// One stored item: the text to return plus the pre-tokenized phrases to match.
struct Entry {
    /// Human-readable content returned on a hit.
    text: String,
    /// Optional title (definition entries); shown as a label in multi-result output.
    title: Option<String>,
    /// Tokenized searchable phrases. Q/A: canonical question + aliases. Doc: one
    /// phrase combining title + body tokens.
    phrases: Vec<Vec<String>>,
    mode: MatchMode,
}

/// A retrieved passage with its relevance score (higher is better).
#[derive(Debug, Clone)]
pub struct Passage {
    pub title: Option<String>,
    pub text: String,
    pub score: f64,
}

/// Local read-only knowledge store with IDF-weighted retrieval.
pub struct KnowledgeStore {
    entries: Vec<Entry>,
    idf: HashMap<String, f64>,
}

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

impl KnowledgeStore {
    /// Load the bundled knowledge plus an optional `data_dir/knowledge.json`
    /// user extension (same shape). Malformed user files are warned about and
    /// skipped rather than failing the whole store.
    pub fn new(data_dir: &Path) -> anyhow::Result<Self> {
        let mut raw: Vec<RawEntry> = serde_json::from_str(BUNDLED_KNOWLEDGE)
            .map_err(|e| anyhow::anyhow!("bundled knowledge.json is invalid: {e}"))?;

        let user = data_dir.join("knowledge.json");
        if user.exists() {
            match std::fs::read_to_string(&user)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str::<Vec<RawEntry>>(&t).map_err(|e| e.to_string()))
            {
                Ok(mut extra) => raw.append(&mut extra),
                Err(e) => eprintln!("knowledge: ignoring {}: {e}", user.display()),
            }
        }

        let mut entries = Vec::with_capacity(raw.len());
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut n_phrases = 0usize;
        let add_phrase = |phrases: &mut Vec<Vec<String>>, toks: Vec<String>,
                              df: &mut HashMap<String, usize>,
                              n_phrases: &mut usize| {
            if toks.is_empty() {
                return;
            }
            *n_phrases += 1;
            for t in toks.iter().collect::<HashSet<_>>() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            phrases.push(toks);
        };

        for r in raw {
            // Q/A entry takes precedence when both an answer and a question exist.
            if let (Some(q), Some(a)) = (r.q.as_ref(), r.a.as_ref()) {
                let mut phrases = Vec::new();
                for p in std::iter::once(q.clone()).chain(r.alt.iter().cloned()) {
                    add_phrase(&mut phrases, tokenize(&p), &mut df, &mut n_phrases);
                }
                if !phrases.is_empty() {
                    entries.push(Entry {
                        text: a.clone(),
                        title: None,
                        phrases,
                        mode: MatchMode::Phrase,
                    });
                }
                continue;
            }
            // Definition / document entry: title + free text.
            if let Some(text) = r.text.as_ref() {
                let mut combined = String::new();
                if let Some(t) = r.title.as_ref() {
                    combined.push_str(t);
                    combined.push(' ');
                }
                combined.push_str(text);
                let mut phrases = Vec::new();
                add_phrase(&mut phrases, tokenize(&combined), &mut df, &mut n_phrases);
                if !phrases.is_empty() {
                    entries.push(Entry {
                        text: text.clone(),
                        title: r.title.clone(),
                        phrases,
                        mode: MatchMode::Query,
                    });
                }
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

    /// Score one entry against the tokenized query. Returns `(score, coverage)`
    /// where `coverage` is the gating ratio (phrase- or query-coverage by mode)
    /// and `score` ranks hits (IDF mass matched, plus coverage as a tie-break).
    fn score_entry(&self, entry: &Entry, q: &[String]) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        let query_total: f64 = q.iter().map(|t| self.idf(t)).sum();
        for phrase in &entry.phrases {
            let (matched, denom) = match entry.mode {
                MatchMode::Phrase => {
                    let total: f64 = phrase.iter().map(|t| self.idf(t)).sum();
                    let matched: f64 = phrase
                        .iter()
                        .filter(|pt| q.iter().any(|qt| tok_match(qt, pt)))
                        .map(|pt| self.idf(pt))
                        .sum();
                    (matched, total)
                }
                MatchMode::Query => {
                    // How much of the QUERY's IDF mass the passage explains.
                    let matched: f64 = q
                        .iter()
                        .filter(|qt| phrase.iter().any(|pt| tok_match(qt, pt)))
                        .map(|qt| self.idf(qt))
                        .sum();
                    (matched, query_total)
                }
            };
            if denom <= 0.0 {
                continue;
            }
            let coverage = matched / denom;
            let score = matched + coverage; // weight first, coverage as tie-break
            if best.map_or(true, |(b, _)| score > b) {
                best = Some((score, coverage));
            }
        }
        best
    }

    /// Top-`k` passages whose coverage clears `min_coverage`, best first. Use
    /// `DEFAULT_MIN_COVERAGE` for recall-oriented RAG, `STRICT_MIN_COVERAGE` for
    /// precision. Returns an empty vec for an empty/all-stopword query.
    pub fn search(&self, query: &str, k: usize, min_coverage: f64) -> Vec<Passage> {
        let q = tokenize(query);
        if q.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut hits: Vec<Passage> = Vec::new();
        for e in &self.entries {
            if let Some((score, coverage)) = self.score_entry(e, &q) {
                if coverage >= min_coverage {
                    hits.push(Passage {
                        title: e.title.clone(),
                        text: e.text.clone(),
                        score,
                    });
                }
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }

    /// Single best Q/A answer above the strict coverage bar — preserves the
    /// historical `lookup` behavior (Q/A entries only, no document passages).
    pub fn best_qa(&self, query: &str) -> Option<&str> {
        let q = tokenize(query);
        if q.is_empty() {
            return None;
        }
        let mut best: Option<(f64, &str)> = None;
        for e in &self.entries {
            if e.mode != MatchMode::Phrase {
                continue;
            }
            if let Some((score, coverage)) = self.score_entry(e, &q) {
                if coverage >= STRICT_MIN_COVERAGE
                    && best.map_or(true, |(b, _)| score > b)
                {
                    best = Some((score, e.text.as_str()));
                }
            }
        }
        best.map(|(_, a)| a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> KnowledgeStore {
        KnowledgeStore::new(Path::new("/nonexistent-tinybit-data")).unwrap()
    }

    #[test]
    fn best_qa_finds_capital_and_blocks_wrong_one() {
        let s = store();
        assert_eq!(s.best_qa("what is the capital of italy?"), Some("Rome is the capital of Italy."));
        // "capital" is shared by many entries; only the country token should win.
        let fr = s.best_qa("what's the capital of france").unwrap();
        assert!(fr.contains("Paris") && !fr.contains("Rome"), "got: {fr}");
    }

    #[test]
    fn best_qa_unknown_is_none() {
        assert!(store().best_qa("airspeed velocity of an unladen swallow").is_none());
        assert!(store().best_qa("who are you").is_none());
    }

    #[test]
    fn search_returns_ranked_passages() {
        let s = store();
        let hits = s.search("capital of japan", 3, DEFAULT_MIN_COVERAGE);
        assert!(!hits.is_empty(), "expected at least one hit");
        assert!(hits[0].text.contains("Tokyo"), "top hit: {}", hits[0].text);
    }

    #[test]
    fn search_empty_query_is_empty() {
        assert!(store().search("who are you", 3, DEFAULT_MIN_COVERAGE).is_empty());
        assert!(store().search("capital of japan", 0, DEFAULT_MIN_COVERAGE).is_empty());
    }
}
