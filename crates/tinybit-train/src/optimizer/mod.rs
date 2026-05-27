// NOTE: these are hand-rolled optimizers retained for reference / future use.
// The trainer currently drives `candle_nn::AdamW` (which integrates with
// candle's autograd backward pass), with manual global L2 gradient clipping
// in `trainer.rs`. Plugging Muon back in requires routing per-tensor grads
// out of candle's `GradStore` for the 2D weight matrices, then back in for
// `optimizer.step(&grads)` — non-trivial; tracked separately.
#[allow(dead_code)]
pub mod adamw;
#[allow(dead_code)]
pub mod muon;

#[allow(unused_imports)]
pub use adamw::AdamW;
#[allow(unused_imports)]
pub use muon::Muon;
