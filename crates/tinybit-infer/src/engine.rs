use crate::sampler::{sample, SamplingParams};
use crate::session::Session;
use candle_core::{Device, DType, Tensor};
use tinybit_core::{config::ModelConfig, model::TinyBit, state::InferenceState, tokenizer::Tokenizer};
use tinybit_tools::ToolRegistry;

pub struct InferenceEngine {
    pub model:     TinyBit,
    pub tokenizer: Tokenizer,
    pub tools:     ToolRegistry,
    pub device:    Device,
    pub params:    SamplingParams,
    /// Token id(s) that begin a `<|tool_call|>` marker. The tool gate bans these
    /// at sampling time to stop the model emitting a tool call on a turn that
    /// doesn't warrant one. Precomputed from the tokenizer once at load.
    pub tool_start_ban: Vec<u32>,
}

/// Timing and token counts for a single generation, surfaced to the CLI.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    /// Tokens fed during prefill (the encoded prompt).
    pub prompt_tokens: usize,
    /// Tokens sampled during the decode loop (one per model forward step).
    pub gen_tokens:    usize,
    /// Wall time spent prefilling the prompt.
    pub prefill_secs:  f64,
    /// Wall time spent in the decode loop (sampling + any tool execution).
    pub decode_secs:   f64,
}

impl GenStats {
    /// Decode throughput in tokens/second.
    pub fn tokens_per_sec(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.gen_tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }

    /// Total wall time for the turn (prefill + decode).
    pub fn total_secs(&self) -> f64 {
        self.prefill_secs + self.decode_secs
    }
}

impl InferenceEngine {
    pub fn new(
        model_path: &std::path::Path,
        config: ModelConfig,
        tokenizer_path: &std::path::Path,
        data_dir: &std::path::Path,
        device: Device,
    ) -> anyhow::Result<Self> {
        // Load the model first — `TinyBit::load` may override config.vocab_size
        // based on the actual embedding shape on disk. Then build the tokenizer
        // capped at the model's true vocab so generated/decoded ids stay in
        // range no matter what's in the user's prompt.
        let model = TinyBit::load(model_path, config, &device)?;
        let tokenizer = Tokenizer::from_file_with_vocab(tokenizer_path, model.config.vocab_size)?;
        let tools = ToolRegistry::with_builtins(data_dir)?;
        // The `<|tool_call|>` marker is BPE text starting with `<`; banning that
        // leading token blocks the whole marker (see processor::ToolMode). But
        // SentencePiece encodes `<` with a DIFFERENT id by context: a
        // space-prefixed `▁<` when it follows a space (or stands alone), and a
        // bare `<` when it follows another char — and the model emits the bare
        // form because it starts right after `assistant:\n`. Ban both, or the
        // gate silently misses (it did at first).
        let call_start = tinybit_tools::parser::CALL_START;
        let mut tool_start_ban: Vec<u32> = Vec::new();
        if let Ok(ids) = tokenizer.encode(call_start, false) {
            if let Some(&id) = ids.first() {
                tool_start_ban.push(id); // ▁< (space-prefixed / standalone)
            }
        }
        if let Ok(ids) = tokenizer.encode(&format!("x{call_start}"), false) {
            if let Some(&id) = ids.get(1) {
                tool_start_ban.push(id); // < (bare, as emitted after a newline)
            }
        }
        tool_start_ban.sort_unstable();
        tool_start_ban.dedup();
        Ok(Self { model, tokenizer, tools, device, params: SamplingParams::default(), tool_start_ban })
    }

    /// Auto-detect best device: Metal on Apple Silicon, CUDA if available, else CPU.
    pub fn auto_device() -> Device {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Ok(dev) = Device::new_metal(0) {
            return dev;
        }
        if candle_core::utils::cuda_is_available() {
            if let Ok(dev) = Device::new_cuda(0) {
                return dev;
            }
        }
        Device::Cpu
    }

    /// Generate tokens until EOS, max_new_tokens, or a stop string is seen
    /// in the decoded output (default: `STOP_STRING_USER_TURN`).
    pub fn generate(
        &self,
        prompt: &str,
        state: &mut InferenceState,
        _on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String> {
        use tinybit_core::tokenizer::STOP_STRING_USER_TURN;
        let ids = self.tokenizer.encode(prompt, false)?;
        let mut generated: Vec<u32> = Vec::new();

        // Prefill all but the last token; the last token's forward in the decode
        // loop yields the first prediction. (Feeding it in both phases would apply
        // it to the recurrent state twice — see ToolProcessor::run.)
        let (prefill_ids, last_id) = match ids.split_last() {
            Some((last, head)) => (head, *last),
            None => (&[][..], self.tokenizer.bos_token_id),
        };
        if !prefill_ids.is_empty() {
            // Chunked sequence prefill — see ToolProcessor::run.
            let ids_t = Tensor::from_vec(prefill_ids.to_vec(), (1, prefill_ids.len()), &self.device)?
                .to_dtype(DType::U32)?;
            self.model.forward_prefill(&ids_t, state)?;
        }
        let mut history: Vec<u32> = ids.clone();

        let mut prev_id = last_id;
        for _ in 0..self.params.max_new_tokens {
            let tid = Tensor::from_vec(vec![prev_id], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            let logits = self.model.forward_step(&tid, state)?;
            let next_id = sample(&logits, &self.params, &history, &[])?;

            if next_id == self.tokenizer.eos_token_id {
                break;
            }
            generated.push(next_id);
            history.push(next_id);
            prev_id = next_id;

            // Decode the tail every few tokens to look for a stop string.
            if generated.len().is_multiple_of(4) {
                let tail = self.tokenizer.decode(&generated, false).unwrap_or_default();
                if tail.contains(STOP_STRING_USER_TURN) {
                    break;
                }
            }
        }

        let mut out = self.tokenizer.decode(&generated, true)?;
        if let Some(idx) = out.find(STOP_STRING_USER_TURN) {
            out.truncate(idx);
        }
        Ok(out.trim_end().to_string())
    }

    /// Process a single chat turn. `tool_mode` gates whether the model may emit
    /// a tool call this turn (see `ToolMode`).
    pub fn chat_turn(
        &self,
        user_message: &str,
        session: &mut Session,
        tool_mode: crate::processor::ToolMode,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<(String, GenStats)> {
        use crate::processor::{message_needs_tools, ToolMode, ToolProcessor};
        let prompt = self.tokenizer.apply_chat_template(
            Some(&session.system_prompt),
            user_message,
        )?;
        // Decide whether to suppress tool emission for this turn.
        let armed = match tool_mode {
            ToolMode::Always => true,
            ToolMode::Never => false,
            ToolMode::Auto => message_needs_tools(user_message),
        };
        let tool_ban: &[u32] = if armed { &[] } else { &self.tool_start_ban };
        let processor = ToolProcessor::new(self);
        let (response, stats) = processor.run(&prompt, &mut session.state, tool_ban, on_token)?;
        session.history.push(crate::session::ChatMessage {
            role: crate::session::Role::User,
            content: user_message.to_string(),
        });
        session.history.push(crate::session::ChatMessage {
            role: crate::session::Role::Assistant,
            content: response.clone(),
        });
        Ok((response, stats))
    }
}
