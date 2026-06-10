use std::path::Path;

/// Plain-text chat template strings. These are never added as special tokens —
/// they are tokenized through the normal vocabulary so the model can be trained
/// on natural text and still understand the role separators.
pub const ROLE_SYSTEM_PREFIX:    &str = "system:\n";
pub const ROLE_USER_PREFIX:      &str = "\nuser:\n";
pub const ROLE_ASSISTANT_PREFIX: &str = "\nassistant:\n";
/// Substring that means "next user turn started" — stop generation when the
/// model emits it.
pub const STOP_STRING_USER_TURN: &str = "\nuser:";

/// Default identity/system prompt for the GENERAL model family. Gives the model
/// its name and persona when no explicit `--system` prompt is set, so it knows
/// it is "tinybit". A base-pretrained model only follows this loosely; reliable
/// persona adherence needs instruction fine-tuning.
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are tinybit, a small and efficient AI assistant built on the RWKV-7 architecture. \
You are helpful, concise, and honest.";

/// Default system prompt for the CODING model family (`*-coding` variants).
/// Steers the assistant toward programming help. Like the general prompt, a
/// base-pretrained model follows it only loosely until instruction-tuned.
pub const CODING_SYSTEM_PROMPT: &str =
    "You are tinybit-coding, a small and efficient coding assistant built on the RWKV-7 \
architecture. You help with programming, Rust, Python, Linux, shell commands, errors, \
and debugging. You are concise, show working code, and are honest about what you are unsure of.";

/// Model family — selects the default system prompt and the kind of help the
/// assistant is tuned for. The architecture is identical across families; the
/// difference is the training-data mix and the default persona.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// General assistant: explanations, notes, todos, summaries, simple Q&A.
    #[default]
    General,
    /// Coding assistant: programming, Rust/Python/Linux, errors, debugging.
    Coding,
}

impl Profile {
    /// Parse a profile name (case-insensitive). Accepts "general" and "coding".
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "general" => Ok(Profile::General),
            "coding" => Ok(Profile::Coding),
            other => anyhow::bail!("unknown profile '{other}' (expected: general | coding)"),
        }
    }

    /// The default system prompt for this family.
    pub fn default_system_prompt(self) -> &'static str {
        match self {
            Profile::General => DEFAULT_SYSTEM_PROMPT,
            Profile::Coding => CODING_SYSTEM_PROMPT,
        }
    }

    /// Best-effort guess from a config path: a filename containing "coding"
    /// (e.g. `configs/micro-coding.toml`) implies the coding family. Lets the
    /// CLI pick a sensible default prompt without an explicit `--profile`.
    pub fn from_config_path(path: &std::path::Path) -> Self {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.to_ascii_lowercase().contains("coding") {
            Profile::Coding
        } else {
            Profile::General
        }
    }
}

/// Tool-call markers. tinybit treats these as ordinary text: data prep
/// tokenizes them as BPE pieces and the model learns that multi-token spelling,
/// so `parse_tool_call` matches on the DECODED text and works regardless of how
/// many tokens a marker spans. They are NOT added to the inference vocabulary
/// (adding them would mint single token ids whose embeddings the model never
/// trained — see `resolve_marker`). The `tool_*_id` fields below are populated
/// only if a tokenizer.json already defines a marker as a real token.
pub const TOOL_CALL_START_STR:   &str = "<|tool_call|>";
pub const TOOL_CALL_END_STR:     &str = "<|end_tool_call|>";
pub const TOOL_RESULT_START_STR: &str = "<|tool_result|>";
pub const TOOL_RESULT_END_STR:   &str = "<|end_tool_result|>";

/// Thin wrapper around `tokenizers::Tokenizer` that knows the *model's* vocab
/// size and is safe to use for inference: encode never returns an id ≥
/// vocab_size, even if the underlying tokenizer was given a literal special-
/// token string in user input.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Maximum id the model can embed (== embedding row count).
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    /// Tool-call markers as token ids — `Some(id)` only when the tokenizer file
    /// already defines the marker as a single token below vocab_size (the
    /// shipped LLaMA tokenizer does not, so these are all `None`). See
    /// `resolve_marker` for why we never mint them ourselves.
    pub tool_call_start_id:   Option<u32>,
    pub tool_call_end_id:     Option<u32>,
    pub tool_result_start_id: Option<u32>,
    pub tool_result_end_id:   Option<u32>,
}

impl Tokenizer {
    /// Resolve a tool marker to a single token id — but ONLY if the tokenizer
    /// file already defines it as a real token below `max_id_exclusive`. We must
    /// NOT *add* it: data prep (`scripts/prepare_data.py`) tokenizes the markers
    /// as ordinary BPE pieces, so the model is trained on that multi-token
    /// spelling and the embedding rows for any freshly-added special id are
    /// untrained noise. (The shipped LLaMA tokenizer has no marker tokens, so
    /// this returns None for all four and encode/decode use the trained BPE
    /// spelling.) A future checkpoint whose tokenizer.json *defines* the markers
    /// — and whose embedding was trained on them — picks up the single-token
    /// fast path automatically. See the regression note below.
    fn resolve_marker(
        tok: &tokenizers::Tokenizer,
        text: &str,
        max_id_exclusive: u32,
    ) -> Option<u32> {
        match tok.token_to_id(text) {
            Some(id) if id < max_id_exclusive => Some(id),
            _ => None,
        }
    }

    fn build(inner: tokenizers::Tokenizer, vocab_size: usize) -> anyhow::Result<Self> {
        let max_id = vocab_size as u32;
        let bos = inner.token_to_id("<s>").unwrap_or(1);
        let eos = inner.token_to_id("</s>").unwrap_or(2);
        let pad = inner.token_to_id("<pad>").unwrap_or(0);

        let tool_call_start   = Self::resolve_marker(&inner, TOOL_CALL_START_STR,   max_id);
        let tool_call_end     = Self::resolve_marker(&inner, TOOL_CALL_END_STR,     max_id);
        let tool_result_start = Self::resolve_marker(&inner, TOOL_RESULT_START_STR, max_id);
        let tool_result_end   = Self::resolve_marker(&inner, TOOL_RESULT_END_STR,   max_id);

        Ok(Self {
            inner,
            vocab_size,
            bos_token_id: bos,
            eos_token_id: eos,
            pad_token_id: pad,
            tool_call_start_id:   tool_call_start,
            tool_call_end_id:     tool_call_end,
            tool_result_start_id: tool_result_start,
            tool_result_end_id:   tool_result_end,
        })
    }

    /// Load from tokenizer.json with vocab_size taken from the file itself.
    /// Use [`from_file_with_vocab`] when the model embedding is smaller than
    /// the tokenizer vocabulary (typical: LLaMA tokenizer + 32000-row embed).
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
        let vocab_size = inner.get_vocab_size(true);
        Self::build(inner, vocab_size)
    }

    /// Load from tokenizer.json, capping ids at `vocab_size`.
    pub fn from_file_with_vocab(path: &Path, vocab_size: usize) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
        Self::build(inner, vocab_size)
    }

    /// Encode `text` to token ids. Any id ≥ `vocab_size` is dropped — this
    /// protects callers from out-of-range index_select crashes when user input
    /// happens to contain a literal special-token string the model wasn't
    /// trained to embed.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        let max = self.vocab_size as u32;
        let before = ids.len();
        ids.retain(|&id| id < max);
        let dropped = before - ids.len();
        if dropped > 0 {
            tracing::warn!(
                vocab_size = self.vocab_size,
                dropped,
                "tokenizer dropped {dropped} token id(s) >= vocab_size during encode"
            );
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))
    }

    /// Start an incremental decoder: feed ids one at a time with
    /// [`IncrementalDecoder::step`]; the concatenation of the yielded chunks
    /// equals `decode(&all_ids, skip_special_tokens)` (pinned by
    /// `test_incremental_decoder_matches_full_decode`). Each step decodes only
    /// a small sliding window of ids, so a generation loop that needs the
    /// decoded-so-far text every token pays O(1) amortized per token instead
    /// of re-decoding the whole buffer (O(n²) over a turn).
    pub fn decode_stream(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tok: self,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Model-visible vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns `true` when the four `<|tool_*|>` markers exist as single-token
    /// ids the model can embed.
    pub fn supports_tool_tokens(&self) -> bool {
        self.tool_call_start_id.is_some()
            && self.tool_call_end_id.is_some()
            && self.tool_result_start_id.is_some()
            && self.tool_result_end_id.is_some()
    }

    /// Plain-text chat template that always tokenizes within `[0, vocab_size)`.
    ///
    /// ```text
    /// system:
    /// {system}
    /// user:
    /// {user}
    /// assistant:
    /// ```
    pub fn apply_chat_template(
        &self,
        system: Option<&str>,
        user: &str,
    ) -> anyhow::Result<Vec<u32>> {
        let mut text = String::new();
        if let Some(sys) = system.filter(|s| !s.is_empty()) {
            text.push_str(ROLE_SYSTEM_PREFIX);
            text.push_str(sys);
        }
        text.push_str(ROLE_USER_PREFIX);
        text.push_str(user);
        text.push_str(ROLE_ASSISTANT_PREFIX);
        self.encode(&text, false)
    }
}

/// Streaming decoder over a [`Tokenizer`] (see [`Tokenizer::decode_stream`]).
///
/// Same sliding-window algorithm as `tokenizers::DecodeStream` (kept here so
/// the vocab-capped wrapper stays the project's only tokenizer surface):
/// decode a window `[carried ids..., new id]`, emit the text that extends the
/// previously decoded prefix, then shrink the window to what the new suffix
/// needs. A step returns `None` while the window's text ends mid-UTF-8
/// (byte-fallback tokens spell one code point over several ids) — the chunk
/// is emitted as soon as a later id completes the code point. Marker/stop
/// scanning stays per-token safe: ASCII-terminated strings like
/// `<|end_tool_call|>` and `\nuser:` can never end mid-code-point, so their
/// completing token always yields its chunk immediately.
pub struct IncrementalDecoder<'a> {
    tok: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl IncrementalDecoder<'_> {
    /// Feed the next id; returns the newly appended text, if any completed.
    pub fn step(&mut self, id: u32) -> anyhow::Result<Option<String>> {
        self.ids.push(id);
        let string = self.tok.decode(&self.ids, self.skip_special_tokens)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{FFFD}') {
            anyhow::ensure!(
                string.starts_with(&self.prefix),
                "incremental decode produced an invalid prefix"
            );
            let new_text = string[self.prefix.len()..].to_string();
            let new_prefix_index = self.ids.len() - self.prefix_index;
            self.ids = self.ids.split_off(self.prefix_index);
            self.prefix = self.tok.decode(&self.ids, self.skip_special_tokens)?;
            self.prefix_index = new_prefix_index;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}
