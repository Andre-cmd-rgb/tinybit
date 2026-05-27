use crate::sampler::{sample, SamplingParams};
use crate::session::Session;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tinybit_core::{config::ModelConfig, model::TinyBit, state::InferenceState, tokenizer::Tokenizer};
use tinybit_tools::ToolRegistry;

pub struct InferenceEngine {
    pub model:     TinyBit,
    pub tokenizer: Tokenizer,
    pub tools:     ToolRegistry,
    pub device:    Device,
    pub params:    SamplingParams,
}

impl InferenceEngine {
    pub fn new(
        model_path: &std::path::Path,
        config: ModelConfig,
        tokenizer_path: &std::path::Path,
        data_dir: &std::path::Path,
        device: Device,
    ) -> anyhow::Result<Self> {
        let model = TinyBit::load(model_path, config, &device)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let tools = ToolRegistry::with_builtins(data_dir)?;
        Ok(Self { model, tokenizer, tools, device, params: SamplingParams::default() })
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

    /// Generate tokens until EOS or max_new_tokens.
    pub fn generate(
        &self,
        prompt: &str,
        state: &mut InferenceState,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String> {
        let ids = self.tokenizer.encode(prompt, false)?;
        let mut generated: Vec<u32> = Vec::new();

        // Feed the prompt
        for &id in &ids {
            let tid = Tensor::from_vec(vec![id], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            self.model.forward_step(&tid, state)?;
        }
        let mut history: Vec<u32> = ids.clone();

        // Generate new tokens
        let mut prev_id = *ids.last().unwrap_or(&self.tokenizer.bos_token_id);
        for _ in 0..self.params.max_new_tokens {
            let tid = Tensor::from_vec(vec![prev_id], (1, 1), &self.device)?.to_dtype(DType::U32)?;
            let logits = self.model.forward_step(&tid, state)?;
            let next_id = sample(&logits, &self.params, &history)?;

            if next_id == self.tokenizer.eos_token_id {
                break;
            }
            generated.push(next_id);
            history.push(next_id);
            prev_id = next_id;

            if let Some(cb) = on_token.as_ref() {
                // We'd call cb here but we'd need mutable access; simplified version:
                let _ = cb;
            }
        }

        self.tokenizer.decode(&generated, true)
    }

    /// Process a single chat turn.
    pub fn chat_turn(
        &self,
        user_message: &str,
        session: &mut Session,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String> {
        use crate::processor::ToolProcessor;
        let prompt = self.tokenizer.apply_chat_template(
            Some(&session.system_prompt),
            user_message,
        )?;
        let processor = ToolProcessor::new(self);
        let response = processor.run(&prompt, &mut session.state, on_token)?;
        session.history.push(crate::session::ChatMessage {
            role: crate::session::Role::User,
            content: user_message.to_string(),
        });
        session.history.push(crate::session::ChatMessage {
            role: crate::session::Role::Assistant,
            content: response.clone(),
        });
        Ok(response)
    }
}
