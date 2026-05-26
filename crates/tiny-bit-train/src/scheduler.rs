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
    pub fn new(peak_lr: f64, total_steps: usize) -> Self {
        let warmup_steps = (total_steps as f64 * 0.02) as usize;
        let decay_steps  = (total_steps as f64 * 0.20) as usize;
        let stable_steps = total_steps - warmup_steps - decay_steps;
        Self {
            peak_lr,
            min_lr: peak_lr * 0.1,
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
