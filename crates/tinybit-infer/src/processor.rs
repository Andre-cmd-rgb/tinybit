use crate::engine::{GenStats, InferenceEngine};
use crate::sampler::sample;
use candle_core::{DType, Tensor};
use std::time::Instant;
use tinybit_core::state::InferenceState;
use tinybit_core::tokenizer::STOP_STRING_USER_TURN;
use tinybit_tools::parser::{format_tool_result, parse_tool_call};

const MAX_TOOL_ROUNDS: usize = 8;

/// Given the visible text decoded so far and how many chars were already
/// streamed, return the next chunk safe to emit and the new emitted count.
/// `holdback` trailing chars are withheld so a partial stop string (`\nuser:`)
/// is never streamed before we get a chance to detect and truncate it; pass
/// `holdback = 0` to flush the remainder once the text is final.
fn next_chunk(visible: &str, emitted: usize, holdback: usize) -> (String, usize) {
    let chars: Vec<char> = visible.chars().collect();
    let safe = chars.len().saturating_sub(holdback);
    if safe > emitted {
        (chars[emitted..safe].iter().collect(), safe)
    } else {
        (String::new(), emitted)
    }
}

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
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<(String, GenStats)> {
        let eng = self.engine;
        let mut full_output = String::new();
        let mut current_ids: Vec<u32> = encoded_prompt.to_vec();
        let mut stats = GenStats { prompt_tokens: encoded_prompt.len(), ..Default::default() };

        // Stream tokens as they're produced when a sink is provided. We hold
        // back the length of the stop string so a partial `\nuser:` is never
        // emitted before the loop detects it and stops.
        let streaming = on_token.is_some();
        let holdback = STOP_STRING_USER_TURN.chars().count();

        let t_prefill = Instant::now();
        for &id in encoded_prompt {
            let tid = Tensor::from_vec(vec![id], (1, 1), &eng.device)?.to_dtype(DType::U32)?;
            eng.model.forward_step(&tid, state)?;
        }
        stats.prefill_secs = t_prefill.elapsed().as_secs_f64();

        let mut prev_id = *encoded_prompt.last().unwrap_or(&eng.tokenizer.bos_token_id);

        let t_decode = Instant::now();
        for _round in 0..self.max_rounds {
            let mut round_tokens: Vec<u32> = Vec::new();
            let mut hit_stop = false;
            let mut emitted = 0usize; // chars of this round already streamed

            for _ in 0..eng.params.max_new_tokens {
                let tid =
                    Tensor::from_vec(vec![prev_id], (1, 1), &eng.device)?.to_dtype(DType::U32)?;
                let logits = eng.model.forward_step(&tid, state)?;
                let next_id = sample(&logits, &eng.params, &current_ids)?;
                stats.gen_tokens += 1;

                if next_id == eng.tokenizer.eos_token_id {
                    let text = eng.tokenizer.decode(&round_tokens, true)?;
                    let trimmed = text.trim_end();
                    if streaming {
                        let (chunk, _) = next_chunk(trimmed, emitted, 0);
                        if let Some(cb) = on_token.as_deref_mut() {
                            cb(&chunk);
                        }
                    }
                    full_output.push_str(trimmed);
                    stats.decode_secs = t_decode.elapsed().as_secs_f64();
                    return Ok((full_output, stats));
                }

                round_tokens.push(next_id);
                current_ids.push(next_id);
                prev_id = next_id;

                if streaming {
                    let visible = eng.tokenizer.decode(&round_tokens, true).unwrap_or_default();
                    let (chunk, new_emitted) = next_chunk(&visible, emitted, holdback);
                    if !chunk.is_empty() {
                        if let Some(cb) = on_token.as_deref_mut() {
                            cb(&chunk);
                        }
                        emitted = new_emitted;
                    }
                }

                // Decode periodically to look for completion conditions.
                if round_tokens.len().is_multiple_of(4) || round_tokens.len() < 8 {
                    let partial =
                        eng.tokenizer.decode(&round_tokens, false).unwrap_or_default();

                    // Complete tool call?
                    if partial.contains("<|end_tool_call|>") {
                        if let Some((call, before, _after)) = parse_tool_call(&partial) {
                            if streaming {
                                let (chunk, _) = next_chunk(before, emitted, 0);
                                if let Some(cb) = on_token.as_deref_mut() {
                                    cb(&chunk);
                                }
                            }
                            full_output.push_str(before);
                            let result = eng.tools.execute(&call).unwrap_or_else(|e| {
                                tinybit_tools::ToolOutput::err(e.to_string())
                            });
                            let result_str = format_tool_result(&result);
                            full_output.push_str(&result_str);
                            if streaming {
                                if let Some(cb) = on_token.as_deref_mut() {
                                    cb(&result_str);
                                }
                            }
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
                let trimmed = text.trim_end();
                if streaming {
                    let (chunk, _) = next_chunk(trimmed, emitted, 0);
                    if let Some(cb) = on_token.as_deref_mut() {
                        cb(&chunk);
                    }
                }
                full_output.push_str(trimmed);
                if hit_stop {
                    break;
                }
                // Otherwise — we hit max_new_tokens without a tool call. Stop.
                break;
            }
        }
        stats.decode_secs = t_decode.elapsed().as_secs_f64();
        Ok((full_output, stats))
    }
}
