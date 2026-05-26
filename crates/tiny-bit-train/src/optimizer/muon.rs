use candle_core::{DType, Tensor};
use std::collections::HashMap;

/// Muon optimizer for 2D weight matrices.
pub struct Muon {
    pub lr:       f64,
    pub momentum: f64,
    pub nesterov: bool,
    pub ns_steps: usize,
    state:        HashMap<String, Tensor>,
}

impl Muon {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            nesterov: true,
            ns_steps: 5,
            state: HashMap::new(),
        }
    }

    /// Newton-Schulz orthogonalization: iteratively contract toward orthogonal matrix.
    fn newton_schulz(x: &Tensor, steps: usize) -> anyhow::Result<Tensor> {
        let norm = x.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        if norm == 0.0 {
            return Ok(x.clone());
        }
        let mut y = (x / norm)?;
        for _ in 0..steps {
            // X = 1.5*X - 0.5 * X @ X.T @ X
            let xt = y.t()?;
            let xtx = y.matmul(&xt.matmul(&y)?)?;
            y = ((y * 1.5_f64)? - (xtx * 0.5_f64)?)?;
        }
        Ok(y)
    }

    /// Update a list of (name, weight, grad) triples.
    /// Weight and grad must be 2D tensors.
    pub fn step(
        &mut self,
        params: &mut [(&str, &mut Tensor, &Tensor)],
    ) -> anyhow::Result<()> {
        for (name, weight, grad) in params.iter_mut() {
            let g = grad.to_dtype(DType::F32)?;
            let m = if let Some(prev_m) = self.state.get(*name) {
                (prev_m * self.momentum)?.add(&(g * (1.0 - self.momentum))?)?
            } else {
                (g * (1.0 - self.momentum))?
            };
            self.state.insert(name.to_string(), m.clone());

            let grad_for_ns = if self.nesterov {
                (m.clone() * self.momentum)?.add(&(grad.to_dtype(DType::F32)? * (1.0 - self.momentum))?)?
            } else {
                m.clone()
            };

            let o = Self::newton_schulz(&grad_for_ns, self.ns_steps)?;
            let update = (o * self.lr)?;
            **weight = weight.to_dtype(DType::F32)?.sub(&update)?;
        }
        Ok(())
    }
}
