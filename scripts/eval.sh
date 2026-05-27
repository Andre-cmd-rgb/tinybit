#!/bin/bash
# Run evaluation suite.
# Usage: ./scripts/eval.sh [model_path] [config_path]
set -euo pipefail

MODEL="${1:-models/tinybit-micro.safetensors}"
CONFIG="${2:-configs/micro.toml}"

echo "Running cargo tests..."
cargo test --workspace 2>&1

echo "Done."
