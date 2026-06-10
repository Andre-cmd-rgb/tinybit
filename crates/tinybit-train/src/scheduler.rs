use std::f64::consts::PI;

/// Warmup-Stable-Decay learning rate schedule.
pub struct WsdScheduler {
    pub peak_lr:      f64,
    pub min_lr:       f64,
    pub warmup_steps: usize,
    pub stable_steps: usize,
    pub total_steps:  usize,
}

impl WsdScheduler {
    /// The validated default shape: 2% linear warmup, 78% stable, 20% cosine
    /// decay to a 10% floor. The documented loss targets (TRAINING.md) assume
    /// this — change shapes only via [`Self::with_shape`] and revalidate per
    /// CLAUDE.md decision 3.
    pub fn new(peak_lr: f64, total_steps: usize) -> Self {
        Self::with_shape(peak_lr, total_steps, 0.02, 0.20, 0.1)
    }

    /// WSD with an explicit shape: `warmup_frac` of steps in linear warmup,
    /// `decay_frac` in cosine decay (the remainder is the stable plateau),
    /// decaying to `peak_lr * min_lr_frac`.
    pub fn with_shape(
        peak_lr: f64,
        total_steps: usize,
        warmup_frac: f64,
        decay_frac: f64,
        min_lr_frac: f64,
    ) -> Self {
        let warmup_steps = (total_steps as f64 * warmup_frac) as usize;
        let decay_steps  = (total_steps as f64 * decay_frac) as usize;
        let stable_steps = total_steps.saturating_sub(warmup_steps + decay_steps);
        Self {
            peak_lr,
            min_lr: peak_lr * min_lr_frac,
            warmup_steps,
            stable_steps: warmup_steps + stable_steps,
            total_steps,
        }
    }

    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            self.peak_lr * (step as f64 / self.warmup_steps as f64)
        } else if step < self.stable_steps {
            // Constant
            self.peak_lr
        } else {
            // Cosine decay from peak to min
            let progress = (step - self.stable_steps) as f64
                / (self.total_steps - self.stable_steps) as f64;
            let progress = progress.min(1.0);
            self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new` must stay bit-identical to `with_shape` at the documented
    /// defaults — the validated runs depend on it.
    #[test]
    fn new_equals_with_shape_defaults() {
        let total = 25_000usize;
        let a = WsdScheduler::new(3e-4, total);
        let b = WsdScheduler::with_shape(3e-4, total, 0.02, 0.20, 0.1);
        assert_eq!(a.warmup_steps, b.warmup_steps);
        assert_eq!(a.stable_steps, b.stable_steps);
        assert_eq!(a.min_lr, b.min_lr);
        for step in [0, 1, 499, 500, 501, 12_500, 19_999, 20_000, 22_500, 24_999, 25_000] {
            assert_eq!(a.get_lr(step), b.get_lr(step), "step {step}");
        }
    }

    #[test]
    fn shape_boundaries() {
        let s = WsdScheduler::with_shape(1.0, 1000, 0.10, 0.20, 0.1);
        assert_eq!(s.warmup_steps, 100);
        assert_eq!(s.stable_steps, 800); // warmup + stable plateau end
        assert_eq!(s.get_lr(0), 0.0);
        assert!((s.get_lr(50) - 0.5).abs() < 1e-12); // mid-warmup
        assert_eq!(s.get_lr(100), 1.0); // plateau start
        assert_eq!(s.get_lr(799), 1.0); // plateau end
        assert!((s.get_lr(1000) - 0.1).abs() < 1e-12); // floor
        // Decay is monotonically non-increasing.
        let mut prev = s.get_lr(800);
        for step in 801..=1000 {
            let lr = s.get_lr(step);
            assert!(lr <= prev + 1e-15, "decay not monotonic at {step}");
            prev = lr;
        }
    }

    /// Degenerate shapes must not underflow/panic.
    #[test]
    fn oversized_fracs_saturate() {
        let s = WsdScheduler::with_shape(1.0, 100, 0.8, 0.8, 0.0);
        assert_eq!(s.stable_steps, s.warmup_steps); // no plateau
        let _ = s.get_lr(0);
        let _ = s.get_lr(99);
        let _ = s.get_lr(100);
    }
}
