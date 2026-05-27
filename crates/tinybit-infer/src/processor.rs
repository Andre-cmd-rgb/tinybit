use crate::engine::InferenceEngine;
use crate::sampler::sample;
use candle_core::{DType, Tensor};
use tinybit_core::state::InferenceState;
use tinybit_core::tokenizer::STOP_STRING_USER_TURN;
use tinybit_tools::parser::{format_tool_result, parse_tool_call};

const MAX_TOOL_ROUNDS: usize = 8;

pub struct ToolProcessor<'a> {
    engine: &'a InferenceEngine,
    max_rounds: usize,
}

impl<'a> ToolProcessor<'a> {
    pub fn new(engine: &'a InferenceEngine) -> Self {
        Self { engine, max_rounds: MAX_TOOL_ROUNDS }
    }

    pub fn run(
        &self,
        encoded_prompt: &[u32],
        state: &mut InferenceState,
        _on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String> {
        let eng = self.engine;
        let mut full_output = String::new();
        let mut current_ids: Vec<u32> = encoded_prompt.to_vec();

        for &id in encoded_prompt {
            let tid = Tensor::from_vec(vec![id], (1, 1), &eng.device)?.to_dtype(DType::U32)?;
            eng.model.forward_step(&tid, state)?;
        }

        let mut prev_id = *encoded_prompt.last().unwrap_or(&eng.tokenizer.bos_token_id);

        for _round in 0..self.max_rounds {
            let mut round_tokens: Vec<u32> = Vec::new();
            let mut hit_stop = false;

            for _ in 0..eng.params.max_new_tokens {
                let tid =
                    Tensor::from_vec(vec![prev_id], (1, 1), &eng.device)?.to_dtype(DType::U32)?;
                let logits = eng.model.forward_step(&tid, state)?;
                let next_id = sample(&logits, &eng.params, &current_ids)?;

                if next_id == eng.tokenizer.eos_token_id {
                    let text = eng.tokenizer.decode(&round_tokens, true)?;
                    full_output.push_str(text.trim_end());
                    return Ok(full_output);
                }

                round_tokens.push(next_id);
                current_ids.push(next_id);
                prev_id = next_id;

                // Decode periodically to look for completion conditions.
                if round_tokens.len().is_multiple_of(4) || round_tokens.len() < 8 {
                    let partial =
                        eng.tokenizer.decode(&round_tokens, false).unwrap_or_default();

                    // Complete tool call?
                    if partial.contains("<|end_tool_call|>") {
                        if let Some((call, before, _after)) = parse_tool_call(&partial) {
                            full_output.push_str(before);
                            let result = eng.tools.execute(&call).unwrap_or_else(|e| {
                                tinybit_tools::ToolOutput::err(e.to_string())
                            });
                            let result_str = format_tool_result(&result);
                            full_output.push_str(&result_str);
                            // Inject result tokens back into context (encode is
                            // vocab-safe and drops any overflow ids).
                            let result_ids = eng.tokenizer.encode(&result_str, false)?;
                            for &rid in &result_ids {
                                let tid = Tensor::from_vec(vec![rid], (1, 1), &eng.device)?
                                    .to_dtype(DType::U32)?;
                                eng.model.forward_step(&tid, state)?;
                                current_ids.push(rid);
                            }
                            prev_id = *result_ids.last().unwrap_or(&prev_id);
                            round_tokens.clear();
                            break; // next round
                        }
                    }

                    // Next user turn marker → stop.
                    if partial.contains(STOP_STRING_USER_TURN) {
                        hit_stop = true;
                        break;
                    }
                }
            }

            if !round_tokens.is_empty() {
                let mut text = eng.tokenizer.decode(&round_tokens, true)?;
                if let Some(idx) = text.find(STOP_STRING_USER_TURN) {
                    text.truncate(idx);
                }
                full_output.push_str(text.trim_end());
                if hit_stop {
                    break;
                }
                // Otherwise — we hit max_new_tokens without a tool call. Stop.
                break;
            }
        }
        Ok(full_output)
    }
}
