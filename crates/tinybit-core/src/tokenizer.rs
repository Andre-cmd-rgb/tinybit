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

/// Default identity/system prompt. Gives the model its name and persona when no
/// explicit `--system` prompt is set, so it knows it is "tinybit". A base-
/// pretrained model only follows this loosely; reliable persona adherence needs
/// instruction fine-tuning.
pub const DEFAULT_SYSTEM_PROMPT: &str =
    "You are tinybit, a small and efficient AI assistant built on the RWKV-7 architecture. \
You are helpful, concise, and honest.";

/// Tool-call markers. These are *only* installed as actual special tokens when
/// the model's vocabulary has room for them (vocab_size > base LLaMA vocab).
/// When that's not the case, the same markers can still be emitted/parsed as
/// plain text — they just tokenize to several normal tokens each, which is fine
/// because `parse_tool_call` matches on decoded text.
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
    /// Tool-call markers as token ids — `Some(id)` only when the marker was
    /// installable as a single special token *and* fits below vocab_size.
    pub tool_call_start_id:   Option<u32>,
    pub tool_call_end_id:     Option<u32>,
    pub tool_result_start_id: Option<u32>,
    pub tool_result_end_id:   Option<u32>,
}

impl Tokenizer {
    fn try_add_special(
        tok: &mut tokenizers::Tokenizer,
        text: &str,
        max_id_exclusive: u32,
    ) -> Option<u32> {
        if let Some(id) = tok.token_to_id(text) {
            return if id < max_id_exclusive { Some(id) } else { None };
        }
        let _added = tok.add_special_tokens(&[tokenizers::AddedToken::from(
            text.to_string(),
            true,
        )]);
        match tok.token_to_id(text) {
            Some(id) if id < max_id_exclusive => Some(id),
            _ => None,
        }
    }

    fn build(mut inner: tokenizers::Tokenizer, vocab_size: usize) -> anyhow::Result<Self> {
        let max_id = vocab_size as u32;
        let bos = inner.token_to_id("<s>").unwrap_or(1);
        let eos = inner.token_to_id("</s>").unwrap_or(2);
        let pad = inner.token_to_id("<pad>").unwrap_or(0);

        let tool_call_start   = Self::try_add_special(&mut inner, TOOL_CALL_START_STR,   max_id);
        let tool_call_end     = Self::try_add_special(&mut inner, TOOL_CALL_END_STR,     max_id);
        let tool_result_start = Self::try_add_special(&mut inner, TOOL_RESULT_START_STR, max_id);
        let tool_result_end   = Self::try_add_special(&mut inner, TOOL_RESULT_END_STR,   max_id);

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

    /// Download from HuggingFace hub (vocab_size = tokenizer's own).
    pub async fn from_hub(model_id: &str) -> anyhow::Result<Self> {
        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json").await?;
        Self::from_file(&tokenizer_path)
    }

    /// Download from HuggingFace hub, capping vocab.
    pub async fn from_hub_with_vocab(model_id: &str, vocab_size: usize) -> anyhow::Result<Self> {
        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json").await?;
        Self::from_file_with_vocab(&tokenizer_path, vocab_size)
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
