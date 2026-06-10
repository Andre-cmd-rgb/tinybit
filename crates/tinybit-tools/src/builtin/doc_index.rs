//! Local document index for the `lookup` tool: paragraph chunks over the
//! user's `data/docs/*.{md,txt}` files, ranked with BM25. This is the
//! "fetch, don't memorize" answer for the user's OWN material (notes,
//! definitions, project docs) — the curated knowledge base stays the first
//! line for general facts (see `LookupTool::execute` precedence).

use crate::builtin::lookup_tool::{tok_match, tokenize};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Per-file size cap: a runaway file (a log, a binary renamed .txt) must not
/// make every lookup slow.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
/// Paragraphs are merged until a chunk reaches at least this many chars …
const CHUNK_MIN_CHARS: usize = 200;
/// … and a chunk is closed before exceeding this many.
const CHUNK_MAX_CHARS: usize = 800;
/// BM25 parameters (standard defaults).
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// One retrievable chunk of a source document.
pub(crate) struct Chunk {
    /// Path relative to the docs dir (attribution shown to the user).
    pub source: String,
    /// Chunk text as it appears in the file (without the heading line).
    pub text: String,
    /// Tokens of heading + text — what BM25 matches against. The nearest
    /// preceding markdown heading is folded in so "definitions under
    /// headings" match queries that only name the heading.
    tokens: Vec<String>,
}

/// Fingerprint of the docs dir (sorted file list + mtime + size). Cheap to
/// recompute per query; the index is rebuilt only when it changes, so newly
/// dropped files are picked up mid-chat without a watcher.
pub(crate) type Fingerprint = Vec<(PathBuf, SystemTime, u64)>;

pub(crate) struct DocIndex {
    chunks: Vec<Chunk>,
    avg_len: f64,
}

/// List indexable files under `docs_dir` (recursive, .md/.txt, size-capped),
/// sorted for a stable fingerprint. Missing dir → empty (the feature is
/// opt-in by creating the dir).
pub(crate) fn scan_docs_dir(docs_dir: &Path) -> Fingerprint {
    let mut out: Fingerprint = Vec::new();
    let mut stack = vec![docs_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(path);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("md") | Some("txt")
            ) && meta.len() <= MAX_FILE_BYTES
            {
                let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                out.push((path, mtime, meta.len()));
            }
        }
    }
    out.sort();
    out
}

/// Split one document into chunks: paragraphs (blank-line separated) merged
/// greedily to CHUNK_MIN..=CHUNK_MAX chars, carrying the nearest preceding
/// `#` heading (markdown) into each chunk's index tokens.
fn chunk_document(source: &str, text: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut heading = String::new();
    let mut buf = String::new();

    let mut flush = |heading: &str, buf: &mut String, chunks: &mut Vec<Chunk>| {
        let text = buf.trim().to_string();
        buf.clear();
        if text.is_empty() {
            return;
        }
        let mut tokens = tokenize(heading);
        tokens.extend(tokenize(&text));
        if tokens.is_empty() {
            return;
        }
        chunks.push(Chunk { source: source.to_string(), text, tokens });
    };

    for para in text.split("\n\n").flat_map(|p| p.split("\r\n\r\n")) {
        let mut trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }
        // A `#` heading sets the context for what follows. The heading is its
        // first line only — body text on the next line (no blank line between)
        // stays chunk text.
        if trimmed.starts_with('#') {
            flush(&heading, &mut buf, &mut chunks);
            let (head_line, rest) = trimmed.split_once('\n').unwrap_or((trimmed, ""));
            heading = head_line.trim_start_matches('#').trim().to_string();
            trimmed = rest.trim();
            if trimmed.is_empty() {
                continue;
            }
        }
        if !buf.is_empty() && buf.len() + trimmed.len() + 2 > CHUNK_MAX_CHARS {
            flush(&heading, &mut buf, &mut chunks);
        }
        if !buf.is_empty() {
            buf.push_str("\n\n");
        }
        buf.push_str(trimmed);
        if buf.len() >= CHUNK_MIN_CHARS {
            flush(&heading, &mut buf, &mut chunks);
        }
    }
    flush(&heading, &mut buf, &mut chunks);
    chunks
}

impl DocIndex {
    /// Build the index over the fingerprinted files. Unreadable/invalid files
    /// are skipped with a warning — a bad file must never break lookup.
    pub(crate) fn build(docs_dir: &Path, files: &Fingerprint) -> Self {
        let mut chunks = Vec::new();
        for (path, _, _) in files {
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("lookup: skipping {}: {e}", path.display());
                    continue;
                }
            };
            let source = path
                .strip_prefix(docs_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            chunks.extend(chunk_document(&source, &text));
        }
        let avg_len = if chunks.is_empty() {
            1.0
        } else {
            chunks.iter().map(|c| c.tokens.len() as f64).sum::<f64>() / chunks.len() as f64
        };
        Self { chunks, avg_len }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Best chunk for the query under BM25 (with the lookup tool's fuzzy
    /// prefix token matching), or None when the evidence is too weak: a match
    /// needs ≥2 distinct query terms, or a single term rare enough (≤10% of
    /// chunks) to be distinctive on its own. Weak matches return None so the
    /// tool reports "not found" instead of noise.
    pub(crate) fn search(&self, query: &str) -> Option<&Chunk> {
        let q = tokenize(query);
        if q.is_empty() || self.chunks.is_empty() {
            return None;
        }
        let n = self.chunks.len();

        // tf[c][t] = fuzzy term frequency of query term t in chunk c.
        let mut tf = vec![vec![0u32; q.len()]; n];
        let mut df = vec![0usize; q.len()];
        for (ci, chunk) in self.chunks.iter().enumerate() {
            for (qi, qt) in q.iter().enumerate() {
                let count = chunk.tokens.iter().filter(|ct| tok_match(qt, ct)).count() as u32;
                tf[ci][qi] = count;
                if count > 0 {
                    df[qi] += 1;
                }
            }
        }

        let idf: Vec<f64> = df
            .iter()
            .map(|&d| (((n as f64 - d as f64 + 0.5) / (d as f64 + 0.5)) + 1.0).ln())
            .collect();
        let rare_bar = (n / 10).max(1);

        let mut best: Option<(f64, usize)> = None;
        for (ci, chunk) in self.chunks.iter().enumerate() {
            let matched: Vec<usize> = (0..q.len()).filter(|&qi| tf[ci][qi] > 0).collect();
            let strong_single =
                matched.len() == 1 && df[matched[0]] <= rare_bar && q.len() == 1;
            if matched.len() < 2 && !strong_single {
                continue;
            }
            let len_norm = 1.0 - BM25_B + BM25_B * chunk.tokens.len() as f64 / self.avg_len;
            let score: f64 = matched
                .iter()
                .map(|&qi| {
                    let f = tf[ci][qi] as f64;
                    idf[qi] * f * (BM25_K1 + 1.0) / (f + BM25_K1 * len_norm)
                })
                .sum();
            if best.map_or(true, |(b, _)| score > b) {
                best = Some((score, ci));
            }
        }
        best.map(|(_, ci)| &self.chunks[ci])
    }
}

/// Trim a chunk to ~`max_chars`, cutting at a sentence boundary where
/// possible — tool results are re-fed into a tiny model's recurrent state, so
/// they must stay short.
pub(crate) fn trim_to_sentence(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let prefix: String = flat.chars().take(max_chars).collect();
    // Prefer the last sentence end in the window; fall back to the last space.
    let cut = prefix
        .rfind(['.', '!', '?'])
        .map(|i| i + 1)
        .or_else(|| prefix.rfind(' '))
        .unwrap_or(prefix.len());
    let mut out = prefix[..cut].trim_end().to_string();
    if !out.ends_with(['.', '!', '?']) {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tinybit-docidx-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn build(dir: &Path) -> DocIndex {
        DocIndex::build(dir, &scan_docs_dir(dir))
    }

    #[test]
    fn chunker_merges_and_carries_headings() {
        let text = "# Rust ownership\n\nShort para.\n\nAnother short one.\n\n# Borrowing\n\nBorrowing lets you reference data without taking ownership of it.";
        let chunks = chunk_document("notes.md", text);
        assert!(chunks.len() >= 2, "expected ≥2 chunks, got {}", chunks.len());
        // The borrowing chunk must be findable via its heading token.
        let b = chunks
            .iter()
            .find(|c| c.text.contains("reference data"))
            .expect("borrowing chunk");
        assert!(b.tokens.iter().any(|t| t == "borrowing"));
        // Headings are context, not chunk text.
        assert!(!b.text.contains('#'));
    }

    #[test]
    fn chunker_respects_max_chars() {
        let para = "word ".repeat(120); // ~600 chars per paragraph
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = chunk_document("big.txt", &text);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.len() <= CHUNK_MAX_CHARS, "chunk too big: {}", c.text.len());
        }
    }

    #[test]
    fn bm25_ranks_on_topic_chunk_first_with_attribution() {
        let dir = tempdir("rank");
        std::fs::write(
            dir.join("pets.md"),
            "# Cats\n\nCats sleep sixteen hours a day and prefer warm spots near windows.\n\n# Dogs\n\nDogs need daily walks and respond well to consistent training routines.",
        )
        .unwrap();
        std::fs::write(
            dir.join("space.txt"),
            "The launch window opens in October. Fuel loading begins twelve hours before liftoff.",
        )
        .unwrap();
        let idx = build(&dir);
        let hit = idx.search("how much do cats sleep").expect("hit");
        assert!(hit.text.contains("sixteen hours"), "got: {}", hit.text);
        assert_eq!(hit.source, "pets.md");
    }

    #[test]
    fn weak_evidence_returns_none() {
        let dir = tempdir("weak");
        std::fs::write(
            dir.join("a.txt"),
            "The quarterly report covers revenue, churn, and hiring plans for the next period.",
        )
        .unwrap();
        let idx = build(&dir);
        // No query term matches at all.
        assert!(idx.search("photosynthesis chlorophyll").is_none());
        // A single common term must not clear the bar on its own.
        assert!(idx.search("the and of").is_none());
    }

    #[test]
    fn missing_dir_is_empty_not_error() {
        let idx = build(Path::new("/nonexistent-tinybit-docs"));
        assert!(idx.is_empty());
        assert!(idx.search("anything").is_none());
    }

    #[test]
    fn fingerprint_changes_when_file_added() {
        let dir = tempdir("fp");
        std::fs::write(dir.join("one.txt"), "alpha beta gamma delta content here").unwrap();
        let fp1 = scan_docs_dir(&dir);
        assert_eq!(fp1.len(), 1);
        std::fs::write(dir.join("two.md"), "# More\n\nepsilon zeta eta theta").unwrap();
        let fp2 = scan_docs_dir(&dir);
        assert_ne!(fp1, fp2);
        assert_eq!(fp2.len(), 2);
    }

    #[test]
    fn non_text_extensions_and_subdirs() {
        let dir = tempdir("sub");
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(
            dir.join("nested/deep.md"),
            "Mitochondria are the powerhouse of the cell, producing ATP through respiration.",
        )
        .unwrap();
        std::fs::write(dir.join("image.png"), [0u8, 159, 146, 150]).unwrap();
        let idx = build(&dir);
        let hit = idx.search("mitochondria powerhouse").expect("hit in nested file");
        assert_eq!(hit.source, std::path::Path::new("nested").join("deep.md").to_string_lossy());
    }

    #[test]
    fn trim_cuts_at_sentence_boundary() {
        let text = "First sentence here. Second sentence is a bit longer than the first. Third one runs on and on with extra words to push past the limit.";
        let out = trim_to_sentence(text, 60);
        assert!(out.chars().count() <= 61, "too long: {} chars", out.chars().count());
        assert!(out.ends_with('.'), "got: {out}");
        let short = trim_to_sentence("tiny", 60);
        assert_eq!(short, "tiny");
    }
}
