# tinybit model variants

tinybit has **two model families**. Both use the *same* RWKV-7 architecture and
the *same* tokenizer — they differ only in the **training-data mix** and the
**default system prompt**. There are four sizes in each family.

| Family  | Variant        | Params | Arch config                  | Data profile | Trainable on L4? |
|---------|----------------|--------|------------------------------|--------------|------------------|
| General | `nano`         | ~25M   | `configs/nano.toml`          | general      | ✅ yes           |
| General | `micro`        | ~50M   | `configs/micro.toml`         | general      | ✅ yes           |
| General | `small`        | ~258M  | `configs/small.toml`         | general      | ❌ inference-only|
| General | `base`         | ~501M  | `configs/base.toml`          | general      | ❌ inference-only|
| Coding  | `nano-coding`  | ~25M   | `configs/nano-coding.toml`   | coding       | ✅ yes           |
| Coding  | `micro-coding` | ~50M   | `configs/micro-coding.toml`  | coding       | ✅ yes           |
| Coding  | `small-coding` | ~258M  | `configs/small-coding.toml`  | coding       | ❌ inference-only|
| Coding  | `base-coding`  | ~501M  | `configs/base-coding.toml`   | coding       | ❌ inference-only|

> `small`/`base` are architectural presets for loading checkpoints trained on
> bigger hardware — they do not fit a single L4 for training. The L4 launcher
> supports `nano` and `micro` (and their `-coding` siblings).

## What each family is for

**General** (`nano`/`micro`/`small`/`base`) — a concise everyday assistant:
explanations, notes, to-dos, calendar, summaries, simple questions, light tool
use, and local context. General models stay general; they are *not* tuned to be
code-heavy.

**Coding** (`*-coding`) — a programming-focused assistant: Rust, Python, Linux,
shell commands, reading errors, small repositories, and debugging workflows.

## How the families actually differ

The architecture (`*-coding.toml` vs its sibling) is **byte-for-byte identical**.
A checkpoint trained either way loads under either config. The difference is:

1. **Training data** — `scripts/prepare_data.sh` takes `DATA_PROFILE=general|coding`:
   - *general*: FineWeb-Edu + Cosmopedia v2 + TinyStories + chat data + a little code.
   - *coding*: The Stack (Python/Rust/JS/C/Go) + technical chat + some prose.
2. **Default system prompt** — `--profile coding` (or a config filename
   containing `coding`) selects the coding persona at chat/eval time.

## Training a variant

Local smoke test (any size, CPU, minutes):

```bash
tinybit train --model-config configs/nano.toml \
              --train-config configs/train-nano-l4.toml --smoke-test
```

Full run on a GCP L4 (general):

```bash
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
  ./scripts/gcp_launch.sh micro
```

Full run on a GCP L4 (coding — note `HF_TOKEN`, code datasets are gated):

```bash
HF_TOKEN=hf_xxx DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-micro-l4.toml \
  ./scripts/gcp_launch.sh micro-coding
```

The training **hyperparameters are identical** across families — reuse the same
`train-*-l4.toml`. Only the model size config and the data profile change.

## Running a variant

```bash
# general
tinybit chat  --config configs/micro.toml        --model models/tinybit-micro.safetensors
# coding (persona auto-selected from the "coding" config name; --profile to force)
tinybit chat  --config configs/micro-coding.toml  --model models/tinybit-micro-coding.safetensors
tinybit chat  --config configs/micro.toml         --model my.safetensors --profile coding
```

## Status

tinybit V1.0 ships **no pretrained weights** — every variant is something you
train. The pipeline, configs, and prompt format are validated; published
checkpoints are future work.
