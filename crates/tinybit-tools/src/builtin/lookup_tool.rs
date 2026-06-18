use crate::knowledge::{KnowledgeStore, STRICT_MIN_COVERAGE};
use crate::tool::{Tool, ToolOutput};
use std::path::Path;

#[derive(serde::Deserialize)]
struct LookupArgs {
    query: String,
}

/// Local read-only fact lookup. A tiny model can't *store* facts reliably, but
/// it can learn to *fetch* them: this returns the best-matching curated fact for
/// a query, or a clear "not found" so the model doesn't bluff.
///
/// Backed by the shared [`KnowledgeStore`]: it first tries the precise Q/A
/// match (historical behavior), then falls back to the best definition/document
/// passage that clears the strict coverage bar.
pub struct LookupTool {
    store: KnowledgeStore,
}

impl LookupTool {
    pub fn new(data_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self { store: KnowledgeStore::new(data_dir)? })
    }
}

impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn description(&self) -> &str {
        "Look up a fact or definition from the local knowledge base (capitals, geography, science, space, definitions). Use it for factual questions instead of guessing."
    }
    fn args_schema(&self) -> &str {
        r#"{"query":"string"}"#
    }

    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        let parsed: LookupArgs =
            serde_json::from_str(args).map_err(|e| anyhow::anyhow!("invalid args: {e}"))?;
        // Precise Q/A answer first; then the strongest definition passage.
        if let Some(answer) = self.store.best_qa(&parsed.query) {
            return Ok(ToolOutput::ok(answer.to_string()));
        }
        if let Some(p) = self
            .store
            .search(&parsed.query, 1, STRICT_MIN_COVERAGE)
            .into_iter()
            .next()
        {
            return Ok(ToolOutput::ok(p.text));
        }
        Ok(ToolOutput::ok(format!(
            "No local entry for \"{}\".",
            parsed.query.trim()
        )))
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

    #[test]
    fn finds_definition_passage() {
        // Definition entries (title/text) are retrievable alongside Q/A facts.
        let a = ask(&tool(), "what is photosynthesis");
        assert!(a.to_lowercase().contains("photosynthesis"), "got: {a}");
    }
}
