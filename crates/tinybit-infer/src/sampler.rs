use candle_core::Tensor;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    pub temperature:        f64,
    pub top_p:              f64,
    pub top_k:              usize,
    pub max_new_tokens:     usize,
    pub repetition_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            // Low default temperature: tinybit-scale models derail when sampled
            // hot (see ChatArgs::temperature). 0.4 stays coherent.
            temperature: 0.4,
            top_p: 0.9,
            top_k: 0,
            max_new_tokens: 512,
            repetition_penalty: 1.1,
        }
    }
}

/// Sample the next token from logits (1, vocab_size).
///
/// `banned` token ids are forced to probability zero (logit −∞) before greedy
/// or stochastic selection. The tool gate uses this to suppress the token that
/// begins `<|tool_call|>` when a turn shouldn't call a tool (see
/// `processor::message_needs_tools`).
pub fn sample(
    logits: &Tensor,
    params: &SamplingParams,
    token_history: &[u32],
    banned: &[u32],
) -> anyhow::Result<u32> {
    use candle_core::DType;
    let mut logits_v = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    // Repetition penalty
    if params.repetition_penalty != 1.0 {
        for &id in token_history {
            let idx = id as usize;
            if idx < logits_v.len() {
                if logits_v[idx] > 0.0 {
                    logits_v[idx] /= params.repetition_penalty as f32;
                } else {
                    logits_v[idx] *= params.repetition_penalty as f32;
                }
            }
        }
    }

    // Banned tokens (tool gate): make them unselectable. −∞ is never the greedy
    // max and contributes 0 mass to the softmax.
    for &id in banned {
        let idx = id as usize;
        if idx < logits_v.len() {
            logits_v[idx] = f32::NEG_INFINITY;
        }
    }

    // Greedy or sampling
    if params.temperature == 0.0 {
        // Greedy
        let best = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return Ok(best as u32);
    }

    // Temperature
    for v in &mut logits_v {
        *v /= params.temperature as f32;
    }

    // Softmax
    let max_v = logits_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_v.iter().map(|&v| (v - max_v).exp()).collect();

    // Top-k filter
    if params.top_k > 0 {
        let k = params.top_k.min(probs.len());
        let mut sorted = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];
        for p in &mut probs {
            if *p < threshold { *p = 0.0; }
        }
    }

    // Top-p filter
    if params.top_p < 1.0 {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
            sorted_idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            // Keep the smallest set of tokens whose cumulative probability
            // reaches top_p — including the token that crosses the threshold —
            // and zero everything after it. (Zeroing the crossing token too
            // would over-truncate the nucleus.)
            let mut cumsum = 0.0f32;
            let mut reached = false;
            for &idx in &sorted_idx {
                if reached {
                    probs[idx] = 0.0;
                    continue;
                }
                cumsum += probs[idx] / sum;
                if cumsum >= params.top_p as f32 {
                    reached = true;
                }
            }
        }
    }

    // Normalize and sample
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 {
        return Ok(0);
    }
    for p in &mut probs {
        *p /= sum;
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}
