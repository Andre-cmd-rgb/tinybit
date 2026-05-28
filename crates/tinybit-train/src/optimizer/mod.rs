// `muon` is wired into the trainer behind the `optimizer = "muon"` config flag:
// it drives the 2D hidden weight matrices (orthogonalized updates via
// Newton-Schulz) while `candle_nn::AdamW` handles the tied embedding/LM-head,
// norms, and biases. See `apply_muon` in `trainer.rs`.
//
// `adamw` here is a hand-rolled reference implementation retained for future
// use; the trainer drives `candle_nn::AdamW`, which integrates with candle's
// autograd `GradStore`.
#[allow(dead_code)]
pub mod adamw;
pub mod muon;

#[allow(unused_imports)]
pub use adamw::AdamW;
pub use muon::Muon;
