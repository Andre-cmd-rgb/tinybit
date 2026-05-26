use candle_core::{DType, Tensor};
use std::collections::HashMap;

/// AdamW optimizer for 1D params (embeddings, biases, LayerNorm).
pub struct AdamW {
    pub lr:           f64,
    pub beta1:        f64,
    pub beta2:        f64,
    pub eps:          f64,
    pub weight_decay: f64,
    step_count:       usize,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

impl AdamW {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        params: &mut [(&str, &mut Tensor, &Tensor)],
    ) -> anyhow::Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;
        let bc1 = 1.0 - self.beta1.powi(t as i32);
        let bc2 = 1.0 - self.beta2.powi(t as i32);

        for (name, weight, grad) in params.iter_mut() {
            let g = grad.to_dtype(DType::F32)?;
            let m = if let Some(prev) = self.m.get(*name) {
                (prev * self.beta1)?.add(&(g.clone() * (1.0 - self.beta1))?)?
            } else {
                (g.clone() * (1.0 - self.beta1))?
            };
            let v = if let Some(prev) = self.v.get(*name) {
                (prev * self.beta2)?.add(&(g.sqr()? * (1.0 - self.beta2))?)?
            } else {
                (g.sqr()? * (1.0 - self.beta2))?
            };
            self.m.insert(name.to_string(), m.clone());
            self.v.insert(name.to_string(), v.clone());

            let m_hat = (m / bc1)?;
            let v_hat = (v / bc2)?;
            let update = m_hat.div(&(v_hat.sqrt()? + self.eps)?)?;
            let w = weight.to_dtype(DType::F32)?;
            // Weight decay
            let w_decayed = (w.clone() * (1.0 - self.lr * self.weight_decay))?;
            **weight = (w_decayed - (update * self.lr)?)?;
        }
        Ok(())
    }
}
