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

/// Repetition penalty is PER OCCURRENCE: a token seen n times in the history is
/// penalized by `penalty^n` (divide positive logits, multiply negative ones).
/// This matches the original per-token loop exactly (sign never flips under a
/// positive factor) but costs one pass over the history instead of one pass per
/// occurrence. Do NOT "fix" this to HF's once-per-distinct-token semantics —
/// the shipped checkpoints were tuned against per-occurrence behavior.
fn apply_repetition_penalty(logits: &mut [f32], penalty: f64, token_history: &[u32]) {
    if penalty == 1.0 || token_history.is_empty() {
        return;
    }
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &id in token_history {
        if (id as usize) < logits.len() {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    let p = penalty as f32;
    for (&id, &n) in &counts {
        let idx = id as usize;
        let factor = p.powi(n as i32);
        if logits[idx] > 0.0 {
            logits[idx] /= factor;
        } else {
            logits[idx] *= factor;
        }
    }
}

/// Total-order comparator over (prob, index): probability descending, index
/// ascending. The index tie-break makes the order strict, so a
/// `select_nth_unstable_by` partition + sort of the head is IDENTICAL to a
/// stable full sort by descending probability — which is what the original
/// implementation used. (NaN probs cannot occur here: probs come from `exp`,
/// and banned logits are −∞ → exp = 0.)
fn cmp_desc(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.cmp(&b.1))
}

/// Top-k filter: zero every prob strictly below the k-th largest value.
/// Ties AT the threshold are all kept (may leave more than k tokens nonzero) —
/// same semantics as the original full-sort implementation, but found with an
/// O(V) selection instead of an O(V log V) sort.
fn apply_top_k(probs: &mut [f32], k: usize) {
    if k == 0 || k >= probs.len() {
        return;
    }
    let mut pairs: Vec<(f32, u32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect();
    pairs.select_nth_unstable_by(k - 1, cmp_desc);
    let threshold = pairs[k - 1].0;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Nucleus (top-p) filter: walk tokens in descending-probability order (ties by
/// index, matching the original stable sort), keep the smallest set whose
/// cumulative probability reaches `top_p` — INCLUDING the crossing token — and
/// zero everything after it.
///
/// Fast path: the nucleus almost always fits in the few hundred most likely
/// tokens, so partition the top M candidates first and only fall back to wider
/// selections (then a full sort) if the cumulative mass hasn't crossed. Because
/// `cmp_desc` is a strict total order, every tier visits tokens in exactly the
/// order the original full sort did, accumulating the same floats in the same
/// order — the output is bit-identical.
fn apply_top_p(probs: &mut [f32], top_p: f64) {
    if top_p >= 1.0 {
        return;
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return;
    }
    let v = probs.len();
    let mut pairs: Vec<(f32, u32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect();

    let mut m = 256usize.min(v);
    loop {
        if m < v {
            pairs.select_nth_unstable_by(m - 1, cmp_desc);
        }
        pairs[..m].sort_unstable_by(cmp_desc);

        let mut cumsum = 0.0f32;
        let mut crossed_at: Option<usize> = None;
        for (pos, &(p, _)) in pairs[..m].iter().enumerate() {
            cumsum += p / sum;
            if cumsum >= top_p as f32 {
                crossed_at = Some(pos);
                break;
            }
        }

        match crossed_at {
            Some(pos) => {
                // Zero everything after the crossing token: the sorted tail of
                // the head slice plus the whole unsorted remainder.
                for &(_, idx) in &pairs[pos + 1..] {
                    probs[idx as usize] = 0.0;
                }
                return;
            }
            None if m == v => return, // total mass never reached top_p; keep all
            None => m = (m * 4).min(v),
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

    apply_repetition_penalty(&mut logits_v, params.repetition_penalty, token_history);

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

    // Softmax (unnormalized; the final draw normalizes)
    let max_v = logits_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_v.iter().map(|&v| (v - max_v).exp()).collect();

    apply_top_k(&mut probs, params.top_k);
    apply_top_p(&mut probs, params.top_p);

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn logits_tensor(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), (1, vals.len()), &Device::Cpu).expect("tensor")
    }

    #[test]
    fn greedy_picks_argmax() {
        let t = logits_tensor(&[0.1, 3.0, -1.0, 2.9]);
        let params = SamplingParams { temperature: 0.0, ..Default::default() };
        let id = sample(&t, &params, &[], &[]).expect("sample");
        assert_eq!(id, 1);
    }

    #[test]
    fn ban_beats_argmax() {
        let t = logits_tensor(&[0.1, 3.0, -1.0, 2.9]);
        let params = SamplingParams { temperature: 0.0, ..Default::default() };
        let id = sample(&t, &params, &[], &[1]).expect("sample");
        assert_eq!(id, 3); // next-best after the ban
    }

    #[test]
    fn ban_holds_under_sampling() {
        // Token 1 dwarfs everything; if the ban leaked it would always win.
        let t = logits_tensor(&[0.0, 50.0, 0.1, 0.2]);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        for _ in 0..50 {
            let id = sample(&t, &params, &[], &[1]).expect("sample");
            assert_ne!(id, 1);
        }
    }

    #[test]
    fn top_k_keeps_exactly_k_largest() {
        let mut probs = vec![0.05, 0.4, 0.1, 0.3, 0.15];
        apply_top_k(&mut probs, 2);
        assert_eq!(probs, vec![0.0, 0.4, 0.0, 0.3, 0.0]);
    }

    #[test]
    fn top_k_keeps_threshold_ties() {
        // Two tokens tie at the k-th value: both survive (original semantics).
        let mut probs = vec![0.4, 0.3, 0.3, 0.1];
        apply_top_k(&mut probs, 2);
        assert_eq!(probs, vec![0.4, 0.3, 0.3, 0.0]);
    }

    #[test]
    fn top_p_includes_crossing_token() {
        // Descending: 0.5, 0.3, 0.15, 0.05. cumsum: 0.5, 0.8 (crosses 0.7).
        let mut probs = vec![0.15, 0.5, 0.05, 0.3];
        apply_top_p(&mut probs, 0.7);
        assert_eq!(probs, vec![0.0, 0.5, 0.0, 0.3]);
    }

    #[test]
    fn top_p_keeps_all_when_threshold_one() {
        let mut probs = vec![0.15, 0.5, 0.05, 0.3];
        let orig = probs.clone();
        apply_top_p(&mut probs, 1.0);
        assert_eq!(probs, orig);
    }

    #[test]
    fn top_p_fast_path_matches_full_sort() {
        // > 256 tokens so the tiered selection actually engages, with a nucleus
        // both inside the first tier and (second case) spanning past it.
        let v = 2000usize;
        for &(top_p, peaked) in &[(0.5f64, true), (0.999f64, false)] {
            let mut probs: Vec<f32> = (0..v)
                .map(|i| {
                    if peaked && i < 5 {
                        1.0
                    } else {
                        1.0 / (i as f32 + 2.0)
                    }
                })
                .collect();
            // Reference: the original full-stable-sort implementation.
            let mut reference = probs.clone();
            {
                let sum: f32 = reference.iter().sum();
                let mut sorted_idx: Vec<usize> = (0..reference.len()).collect();
                sorted_idx.sort_by(|&a, &b| {
                    reference[b]
                        .partial_cmp(&reference[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut cumsum = 0.0f32;
                let mut reached = false;
                for &idx in &sorted_idx {
                    if reached {
                        reference[idx] = 0.0;
                        continue;
                    }
                    cumsum += reference[idx] / sum;
                    if cumsum >= top_p as f32 {
                        reached = true;
                    }
                }
            }
            apply_top_p(&mut probs, top_p);
            assert_eq!(probs, reference, "top_p={top_p}");
        }
    }

    #[test]
    fn repetition_penalty_is_per_occurrence() {
        // Token 1 appears 3× in history → divided by penalty^3 (positive logit);
        // token 2 appears once with a negative logit → multiplied by penalty.
        let mut logits = vec![1.0f32, 2.0, -1.0];
        apply_repetition_penalty(&mut logits, 1.1, &[1, 2, 1, 1]);
        assert_eq!(logits[0], 1.0);
        assert!((logits[1] - 2.0 / 1.1f32.powi(3)).abs() < 1e-6);
        assert!((logits[2] - (-1.0 * 1.1)).abs() < 1e-6);
    }

    #[test]
    fn all_banned_degenerate_still_returns_in_range() {
        // Every token banned → all logits −∞ → softmax is NaN. The historical
        // (preserved) behavior is the end-of-loop fallback to the last index;
        // the contract worth pinning is "no panic, in-range id".
        let t = logits_tensor(&[1.0, 2.0]);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let id = sample(&t, &params, &[], &[0, 1]).expect("sample");
        assert!(id < 2);
    }

    #[test]
    fn sampling_respects_top_k_support() {
        // With top_k=2 only the two largest logits may ever be drawn.
        let t = logits_tensor(&[0.0, 5.0, 1.0, 4.0, 0.5]);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        for _ in 0..50 {
            let id = sample(&t, &params, &[], &[]).expect("sample");
            assert!(id == 1 || id == 3, "sampled outside top-k: {id}");
        }
    }
}
