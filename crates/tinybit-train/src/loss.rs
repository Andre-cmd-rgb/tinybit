use candle_core::{DType, Tensor};

/// Cross-entropy loss for next-token prediction.
/// logits: (B, T, vocab_size), targets: (B, T) as u32
pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
) -> anyhow::Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    // Flatten to (B*T, vocab) and (B*T,)
    let logits_flat = logits.reshape((b * t, v))?;
    let targets_flat = targets.reshape(b * t)?.to_dtype(DType::U32)?;

    // Log-softmax
    let log_probs = candle_nn::ops::log_softmax(&logits_flat.to_dtype(DType::F32)?, 1)?;

    // Gather log-prob at target indices
    let targets_i64 = targets_flat.to_dtype(DType::I64)?;
    let gathered = log_probs.gather(&targets_i64.unsqueeze(1)?, 1)?.squeeze(1)?;

    // Negative mean loss
    let loss = gathered.neg()?.mean_all()?;
    Ok(loss)
}

/// Memory-frugal cross-entropy for next-token prediction.
///
/// The standard path (`cross_entropy_loss` above) materializes the full
/// `(B*T, vocab)` logits AND its log-softmax in F32 for autograd. At the micro
/// shape that is ~0.7 GB each, retained for backward — and now that the WKV
/// scan is fused, this loss is what pins `batch_size`. This computes the loss
/// and its gradients analytically in row-chunks of `chunk_rows`, so peak extra
/// memory is `(chunk_rows, vocab)` instead of `(B*T, vocab)`.
///
/// The softmax gradient is exact (`p - onehot`), so the returned grads are
/// numerically identical to differentiating
/// `cross_entropy_loss((normed @ weight^T) * scale)` — pinned by the
/// `fused_ce_matches_autograd` test below.
///
///   normed  (B, T, D)  — post-final-LayerNorm hidden states (pre-LM-head)
///   weight  (V, D)     — LM-head / tied-embedding matrix
///   scale              — logit scale (1/sqrt(d_model) in tinybit)
///   targets (B, T) u32 — next-token ids
/// Returns (loss, d_loss/d_normed [B,T,D], d_loss/d_weight [V,D]); the loss is a
/// detached on-device F32 scalar (so the caller syncs it once per step), grads
/// are F32.
pub fn fused_cross_entropy_grads(
    normed: &Tensor,
    weight: &Tensor,
    scale: f64,
    targets: &Tensor,
    chunk_rows: usize,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let (b, t, d) = normed.dims3()?;
    let v = weight.dim(0)?;
    let n = b * t;
    let dev = normed.device();

    // Detach: the chunked matmuls below must NOT build/retain an autograd graph
    // over the (chunk, vocab) logits — that is the whole point (memory). The
    // gradients w.r.t. `normed` and `weight` are returned explicitly and
    // re-injected by the caller (a surrogate `(normed * grad_normed).sum()
    // .backward()` plus a manual `grad_w` merge), so candle must not track ANY of
    // this. BOTH inputs have to be detached: `weight` is a model parameter that
    // requires grad, so a non-detached `w` would make every chunk's logits/
    // softmax `(C, V)` tensors graph nodes — and `g_nf` (pushed into
    // `grad_normed_chunks`) would then pin that graph for EVERY chunk, retaining
    // ~6×(B*T, V) instead of freeing each chunk. That un-detached `weight` was the
    // VRAM regression that made fused CE OOM where standard CE fit (it used MORE
    // memory, not less). Keep both detaches.
    let normed_flat = normed.detach().reshape((n, d))?.to_dtype(DType::F32)?;
    let targets_flat = targets.reshape((n,))?.to_dtype(DType::I64)?;
    let w = weight.detach().to_dtype(DType::F32)?; // (V, D) — detached, see above
    let wt = w.t()?.contiguous()?; // (D, V)

    let mut grad_w = Tensor::zeros((v, d), DType::F32, dev)?;
    let mut grad_normed_chunks: Vec<Tensor> = Vec::new();
    let mut nll_sums: Vec<Tensor> = Vec::new();

    let chunk = chunk_rows.max(1);
    let mut s = 0usize;
    while s < n {
        let len = chunk.min(n - s);
        let nf = normed_flat.narrow(0, s, len)?; // (C, D)
        let idx = targets_flat.narrow(0, s, len)?.unsqueeze(1)?; // (C, 1)

        let logits = (nf.matmul(&wt)? * scale)?; // (C, V) f32
        let maxv = logits.max_keepdim(1)?; // (C, 1)
        let shifted = logits.broadcast_sub(&maxv)?; // (C, V)
        let expv = shifted.exp()?; // (C, V)
        let sumexp = expv.sum_keepdim(1)?; // (C, 1)

        // loss += sum over rows of (logsumexp - shifted[target]) == -log p[target]
        let shifted_tgt = shifted.gather(&idx, 1)?; // (C, 1)
        let nll = (sumexp.log()? - shifted_tgt)?; // (C, 1)
        nll_sums.push(nll.sum_all()?); // kept on-device; summed once below

        // d loss / d logits = (softmax - onehot) / n
        let p = expv.broadcast_div(&sumexp)?; // (C, V)
        let ones = Tensor::ones((len, 1), DType::F32, dev)?;
        let onehot = Tensor::zeros((len, v), DType::F32, dev)?.scatter_add(&idx, &ones, 1)?;
        let g_logits = ((p - onehot)? * (1.0 / n as f64))?; // (C, V)

        // chain through logits = (nf @ w^T) * scale
        let g_nf = (g_logits.matmul(&w)? * scale)?; // (C,V)@(V,D) = (C,D)
        let g_w_chunk = (g_logits.t()?.contiguous()?.matmul(&nf)? * scale)?; // (V,C)@(C,D) = (V,D)
        grad_w = (grad_w + g_w_chunk)?;
        grad_normed_chunks.push(g_nf);

        s += len;
    }

    let grad_normed = Tensor::cat(&grad_normed_chunks, 0)?.reshape((b, t, d))?;
    // Mean NLL over all rows, kept on-device (detached — built from detached
    // `normed`), so the training loop syncs it once per step like the standard path.
    let loss = (Tensor::stack(&nll_sums, 0)?.sum_all()? / n as f64)?;
    Ok((loss, grad_normed, grad_w))
}

/// KL divergence distillation loss.
/// alpha * KL + (1-alpha) * CE
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_log_probs: &Tensor,
    teacher_indices: &Tensor,
    alpha: f64,
) -> anyhow::Result<Tensor> {
    let (b, t, _v) = student_logits.dims3()?;
    let k = teacher_log_probs.dim(2)?;

    let student_flat = student_logits.reshape((b * t, student_logits.dim(2)?))?;
    let student_log = candle_nn::ops::log_softmax(&student_flat.to_dtype(DType::F32)?, 1)?;

    // Gather student log probs at teacher indices
    let idx_flat = teacher_indices.reshape((b * t, k))?.to_dtype(DType::I64)?;
    let student_at_teacher = student_log.gather(&idx_flat, 1)?;

    // KL = sum(teacher_probs * (teacher_log - student_log))
    let teacher_flat = teacher_log_probs.reshape((b * t, k))?;
    let teacher_probs = teacher_flat.exp()?;
    let kl = (teacher_probs * (teacher_flat - student_at_teacher)?)?.sum_keepdim(1)?.mean_all()?;

    // Also compute CE for targets (use argmax of teacher as proxy)
    let targets = teacher_indices.narrow(2, 0, 1)?.squeeze(2)?; // take top-1
    let ce = cross_entropy_loss(student_logits, &targets)?;

    Ok(((kl * alpha)? + (ce * (1.0 - alpha))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    }

    /// The analytic chunked CE must produce loss + gradients numerically
    /// identical to differentiating the full-logits path through candle's
    /// autograd. Run on whatever device the test harness picks.
    fn fused_ce_parity_on(dev: &Device) -> anyhow::Result<()> {
        let (b, t, d, v) = (3usize, 7usize, 16usize, 131usize);
        let scale = 1.0 / (d as f64).sqrt();

        let normed = Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, t, d), dev)?)?;
        let weight = Var::from_tensor(&Tensor::randn(0f32, 1f32, (v, d), dev)?)?;
        let tgt: Vec<u32> = (0..b * t).map(|i| ((i * 13 + 5) % v) as u32).collect();
        let targets = Tensor::from_vec(tgt, (b, t), dev)?;

        // Reference: full (B*T, V) logits through autograd.
        let nf = normed.as_tensor().reshape((b * t, d))?;
        let wt = weight.as_tensor().t()?.contiguous()?;
        let logits = (nf.matmul(&wt)? * scale)?.reshape((b, t, v))?;
        let loss_ref = cross_entropy_loss(&logits, &targets)?;
        let loss_ref_val = loss_ref.to_scalar::<f32>()? as f64;
        let grads = loss_ref.backward()?;
        let g_normed_ref = grads.get(normed.as_tensor()).expect("grad normed").clone();
        let g_weight_ref = grads.get(weight.as_tensor()).expect("grad weight").clone();

        // Fused: chunk_rows=4 forces >1 chunk over the 21 rows.
        let (loss_t, g_normed_f, g_weight_f) =
            fused_cross_entropy_grads(normed.as_tensor(), weight.as_tensor(), scale, &targets, 4)?;
        let loss_f = loss_t.to_scalar::<f32>()? as f64;

        assert!(
            (loss_ref_val - loss_f).abs() < 1e-4,
            "loss mismatch: ref={loss_ref_val} fused={loss_f}"
        );
        let dn = max_abs_diff(&g_normed_ref, &g_normed_f)?;
        let dw = max_abs_diff(&g_weight_ref, &g_weight_f)?;
        assert!(dn < 1e-4, "grad_normed max abs diff = {dn}");
        assert!(dw < 1e-4, "grad_weight max abs diff = {dw}");
        Ok(())
    }

    #[test]
    fn fused_ce_matches_autograd() -> anyhow::Result<()> {
        fused_ce_parity_on(&Device::Cpu)
    }

    /// Regression guard for the VRAM blowup (the reason fused CE OOMed where
    /// standard CE fit): the fused outputs MUST be fully detached from autograd,
    /// because the grads are returned analytically. If `weight` (or `normed`) is
    /// left attached, every chunk's `(C, V)` logits/softmax become retained graph
    /// nodes that `grad_normed_chunks` pins for the whole loop — using MORE memory
    /// than the standard `(B*T, V)` path. Here we differentiate the fused outputs
    /// and assert no gradient flows back to the inputs (i.e. there is no graph).
    #[test]
    fn fused_ce_outputs_are_detached() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let (b, t, d, v) = (2usize, 5usize, 8usize, 17usize);
        let scale = 1.0 / (d as f64).sqrt();

        let normed = Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, t, d), &dev)?)?;
        let weight = Var::from_tensor(&Tensor::randn(0f32, 1f32, (v, d), &dev)?)?;
        let tgt: Vec<u32> = (0..b * t).map(|i| (i % v) as u32).collect();
        let targets = Tensor::from_vec(tgt, (b, t), &dev)?;

        let (_loss, g_normed, g_w) =
            fused_cross_entropy_grads(normed.as_tensor(), weight.as_tensor(), scale, &targets, 3)?;

        // Differentiating any fused output must NOT reach the inputs: the outputs
        // are plain data, not graph nodes. With an un-detached `weight`, backward
        // would populate `weight`'s grad here (and the (C,V) graph would have been
        // retained) — that is the regression this guards.
        let probe = (g_normed.sum_all()? + g_w.sum_all()?)?;
        let bg = probe.backward()?;
        assert!(
            bg.get(weight.as_tensor()).is_none(),
            "fused output retains an autograd graph to `weight` — the un-detached-weight VRAM regression is back"
        );
        assert!(
            bg.get(normed.as_tensor()).is_none(),
            "fused output retains an autograd graph to `normed`"
        );
        Ok(())
    }

    /// End-to-end on a real (tiny) TinyBit: the fused training path
    /// (forward_train_normed → analytic loss/grads → surrogate backward +
    /// tied-weight grad merge) must produce the SAME gradient for EVERY model
    /// parameter as the standard path (forward_train → cross_entropy_loss →
    /// backward). This is the contract the trainer wiring relies on.
    #[test]
    fn fused_ce_wiring_matches_standard_grads() -> anyhow::Result<()> {
        use candle_nn::{VarBuilder, VarMap};
        use tinybit_core::{config::ModelConfig, model::TinyBit};

        let dev = Device::Cpu;
        let cfg = ModelConfig {
            vocab_size: 96,
            num_layers: 2,
            d_model: 32,
            d_ffn: 64,
            num_heads: 2,
            head_dim: 16,
            ternary_ffn: false,
            int8_time: false,
            max_seq_len: 32,
            dropout: 0.0,
            spec_heads: 0,
        };
        let (b, t) = (2usize, 12usize);
        let vocab = cfg.vocab_size as u32;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = TinyBit::new(cfg, vb)?;

        let ids: Vec<u32> =
            (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
        let tgt: Vec<u32> =
            (0..b * t).map(|i| ((i + 7) as u32).wrapping_mul(40503) % vocab).collect();
        let input = Tensor::from_vec(ids, (b, t), &dev)?;
        let targets = Tensor::from_vec(tgt, (b, t), &dev)?;

        // (A) standard path.
        let (logits, _) = model.forward_train(&input)?;
        let grads_a = cross_entropy_loss(&logits, &targets)?.backward()?;

        // (B) fused path.
        let normed = model.forward_train_normed(&input)?;
        let w = model.tied_lm_weight().clone();
        let scale = model.logit_scale();
        let (_loss, g_normed, g_w) =
            fused_cross_entropy_grads(&normed, &w, scale, &targets, 5)?;
        let surrogate = normed.broadcast_mul(&g_normed.detach())?.sum_all()?;
        let mut grads_b = surrogate.backward()?;
        let merged = match grads_b.remove(&w) {
            Some(p) => (p + &g_w)?,
            None => g_w.clone(),
        };
        grads_b.insert(&w, merged);

        // Every parameter's gradient must agree between the two paths.
        let data = varmap.data().lock().unwrap();
        let mut worst = 0f32;
        let mut compared = 0usize;
        for (name, v) in data.iter() {
            match (grads_a.get(v.as_tensor()), grads_b.get(v.as_tensor())) {
                (Some(a), Some(b)) => {
                    let d = max_abs_diff(a, b)?;
                    worst = worst.max(d);
                    compared += 1;
                    assert!(d < 2e-4, "param {name}: grad max abs diff = {d}");
                }
                (None, None) => {}
                (a, _) => panic!("param {name}: only one path has a grad (a={})", a.is_some()),
            }
        }
        assert!(compared > 0 && worst.is_finite(), "no gradients compared");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_fused_ce_matches_autograd() -> anyhow::Result<()> {
        fused_ce_parity_on(&Device::new_cuda(0)?)
    }
}
