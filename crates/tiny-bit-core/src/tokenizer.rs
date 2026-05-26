use anyhow::Context;
use std::path::Path;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub bos_token_id:         u32,
    pub eos_token_id:         u32,
    pub pad_token_id:         u32,
    pub tool_call_start_id:   u32,
    pub tool_call_end_id:     u32,
    pub tool_result_start_id: u32,
    pub tool_result_end_id:   u32,
    pub assistant_token_id:   u32,
    pub user_token_id:        u32,
    pub system_token_id:      u32,
}

impl Tokenizer {
    fn get_or_add_special(tok: &mut tokenizers::Tokenizer, text: &str) -> u32 {
        if let Some(id) = tok.token_to_id(text) {
            return id;
        }
        // Add special token — use a placeholder id based on current vocab size
        // (proper addition is done below with add_special_tokens)
        tok.add_special_tokens(&[tokenizers::AddedToken::from(text.to_string(), true)]);
        tok.token_to_id(text).unwrap_or(0)
    }

    fn build(mut inner: tokenizers::Tokenizer) -> anyhow::Result<Self> {
        let bos = inner.token_to_id("<s>").unwrap_or(1);
        let eos = inner.token_to_id("</s>").unwrap_or(2);
        let pad = inner.token_to_id("<pad>").unwrap_or(0);

        let tool_call_start   = Self::get_or_add_special(&mut inner, "<|tool_call|>");
        let tool_call_end     = Self::get_or_add_special(&mut inner, "<|end_tool_call|>");
        let tool_result_start = Self::get_or_add_special(&mut inner, "<|tool_result|>");
        let tool_result_end   = Self::get_or_add_special(&mut inner, "<|end_tool_result|>");
        let assistant_id      = Self::get_or_add_special(&mut inner, "<|assistant|>");
        let user_id           = Self::get_or_add_special(&mut inner, "<|user|>");
        let system_id         = Self::get_or_add_special(&mut inner, "<|system|>");

        Ok(Self {
            inner,
            bos_token_id:         bos,
            eos_token_id:         eos,
            pad_token_id:         pad,
            tool_call_start_id:   tool_call_start,
            tool_call_end_id:     tool_call_end,
            tool_result_start_id: tool_result_start,
            tool_result_end_id:   tool_result_end,
            assistant_token_id:   assistant_id,
            user_token_id:        user_id,
            system_token_id:      system_id,
        })
    }

    /// Load from a saved tokenizer.json.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
        Self::build(inner)
    }

    /// Download from HuggingFace hub.
    pub async fn from_hub(model_id: &str) -> anyhow::Result<Self> {
        let api = hf_hub::api::tokio::Api::new()?;
        let repo = api.model(model_id.to_string());
        let tokenizer_path = repo.get("tokenizer.json").await?;
        Self::from_file(&tokenizer_path)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Format a chat message into the prompt template.
    /// Template: <|system|>{system}<|user|>{user}<|assistant|>
    pub fn apply_chat_template(
        &self,
        system: Option<&str>,
        user: &str,
    ) -> anyhow::Result<Vec<u32>> {
        let mut text = String::new();
        if let Some(sys) = system {
            text.push_str("<|system|>");
            text.push_str(sys);
        }
        text.push_str("<|user|>");
        text.push_str(user);
        text.push_str("<|assistant|>");
        self.encode(&text, false)
    }
}
