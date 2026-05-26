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
