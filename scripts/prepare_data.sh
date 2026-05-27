#!/bin/bash
# Usage: ./scripts/prepare_data.sh [output_dir]
# Downloads and tokenizes datasets for training tinybit.
#
# Env vars:
#   TOTAL_TOKENS   total desired tokens (default 500_000_000)
#   MIN_TOKENS     fail if fewer than this are collected (default 75% of TOTAL_TOKENS)
#   SEQ_LEN        for val-set sizing (default 1024)
#   HF_TOKEN       optional — enables gated datasets (the-stack-smol, etc.)
#   ENABLE_GATED   1 to attempt gated datasets when HF_TOKEN is set (default 1)
#
# Datasets (all opened in streaming mode):
#   FineWeb-Edu, Wikipedia EN, OpenHermes-2.5, dolphin-r1 (nonreasoning), the-stack-smol (gated).
#
# Failure handling:
#   - A dataset that errors does not abort the run; its remaining budget is
#     redistributed proportionally to the surviving datasets.
#   - If the final collected total is below MIN_TOKENS the script exits non-zero
#     so the caller can avoid burning GPU time on a starved dataset.

set -Eeuo pipefail

OUTPUT_DIR="${1:-data}"
mkdir -p "$OUTPUT_DIR"

echo "Preparing data in $OUTPUT_DIR ..."

python3 - "$OUTPUT_DIR" <<'PYTHON'
import os, sys, struct, random
from pathlib import Path

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"

try:
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tqdm import tqdm
except ImportError:
    print("ERROR: Install required packages: pip install datasets tokenizers tqdm")
    sys.exit(1)

SEQ_LEN       = int(os.environ.get("SEQ_LEN", "1024"))
TOTAL_TOKENS  = int(os.environ.get("TOTAL_TOKENS", "500000000"))
MIN_TOKENS    = int(os.environ.get("MIN_TOKENS", str(int(TOTAL_TOKENS * 0.75))))
HF_TOKEN      = os.environ.get("HF_TOKEN", "").strip() or None
ENABLE_GATED  = os.environ.get("ENABLE_GATED", "1") == "1"

print(f"TOTAL_TOKENS={TOTAL_TOKENS:,}  MIN_TOKENS={MIN_TOKENS:,}  HF_TOKEN={'set' if HF_TOKEN else 'unset'}")

# Tokenizer ------------------------------------------------------------------
print("Loading tokenizer...")
try:
    tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
except Exception:
    if os.path.exists("tokenizer.json"):
        tokenizer = Tokenizer.from_file("tokenizer.json")
    else:
        print("ERROR: No tokenizer found. Run: tinybit download --model small")
        sys.exit(1)

EOS_ID = tokenizer.token_to_id("</s>") or 2

# Dataset table --------------------------------------------------------------
# (name, config, split, text_field, weight, gated)
DATASETS = [
    ("HuggingFaceFW/fineweb-edu",         "sample-10BT",  "train", "text",          0.40, False),
    ("wikimedia/wikipedia",               "20231101.en",  "train", "text",          0.30, False),
    ("teknium/OpenHermes-2.5",            None,           "train", "conversations", 0.15, False),
    ("cognitivecomputations/dolphin-r1",  "nonreasoning", "train", "messages",      0.10, False),
    ("bigcode/the-stack-smol",            "data/python",  "train", "content",       0.05, True),
]

def text_from_example(ex, field):
    val = ex.get(field, "")
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                for k in ("value", "content", "text", "output"):
                    if k in item and item[k]:
                        parts.append(str(item[k]))
                        break
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(val)

def load_stream(name, config, split):
    kwargs = {"split": split, "streaming": True}
    if config is not None:
        return load_dataset(name, config, **kwargs)
    return load_dataset(name, **kwargs)

# Skip gated datasets if no auth ---------------------------------------------
active, skipped = [], []
for entry in DATASETS:
    name, cfg, split, field, weight, gated = entry
    if gated and (not HF_TOKEN or not ENABLE_GATED):
        skipped.append((name, "gated and HF_TOKEN missing/disabled"))
        continue
    active.append(entry)

if skipped:
    print("Skipping datasets:")
    for n, reason in skipped:
        print(f"  - {n}: {reason}")

if not active:
    print("ERROR: no datasets remain active.")
    sys.exit(2)

# Token bookkeeping ----------------------------------------------------------
token_buffer = []
collected = {}
remaining_budget = TOTAL_TOKENS
remaining_active = list(active)

while remaining_active:
    weight_sum = sum(w for *_, w, _ in remaining_active)
    name, cfg, split, field, weight, _gated = remaining_active.pop(0)
    target = int(remaining_budget * (weight / weight_sum)) if weight_sum > 0 else 0
    if target <= 0:
        continue
    print(f"  {name}: target {target:,} tokens (cfg={cfg})")
    got = 0
    try:
        ds = load_stream(name, cfg, split)
        for ex in tqdm(ds, desc=name, unit="ex", mininterval=2.0,
                       total=max(target // 512, 100)):
            text = text_from_example(ex, field)
            if not text or not text.strip():
                continue
            try:
                enc = tokenizer.encode(text)
            except Exception:
                continue
            ids = enc.ids + [EOS_ID]
            token_buffer.extend(ids)
            got += len(ids)
            if got >= target:
                break
        collected[name] = got
        remaining_budget = max(0, remaining_budget - got)
        print(f"    {name}: collected {got:,} tokens (remaining budget {remaining_budget:,})")
    except Exception as e:
        print(f"  WARNING: {name} failed: {e}")
        collected[name] = 0
        # Do not decrement remaining_budget — surviving datasets absorb the gap
        # automatically via the proportional split on the next iteration.
        continue

total_collected = sum(collected.values())
print()
print("Per-dataset collection:")
for n, t in collected.items():
    print(f"  {n}: {t:,}")
print(f"Total collected: {total_collected:,}  (target {TOTAL_TOKENS:,})")

if total_collected < MIN_TOKENS:
    print(f"ERROR: only {total_collected:,} tokens collected; MIN_TOKENS={MIN_TOKENS:,}.")
    sys.exit(3)

random.seed(0xB17B17)
random.shuffle(token_buffer)

val_size = max(SEQ_LEN * 100, len(token_buffer) // 50)
val_tokens   = token_buffer[:val_size]
train_tokens = token_buffer[val_size:]

def write_bin(tokens, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        CHUNK = 1 << 20  # write 1M u32s at a time to keep memory bounded
        i = 0
        while i < len(tokens):
            j = min(i + CHUNK, len(tokens))
            f.write(struct.pack(f"<{j-i}I", *tokens[i:j]))
            i = j
    print(f"  Wrote {len(tokens):,} tokens to {path}")

write_bin(train_tokens, f"{OUTPUT_DIR}/train.bin")
write_bin(val_tokens,   f"{OUTPUT_DIR}/val.bin")
print("Done!")
PYTHON
