#!/bin/bash
# Usage: ./scripts/prepare_data.sh [output_dir]
# Downloads and tokenizes datasets for training tiny-bit.
#
# Datasets (all free, no paid APIs):
#   - FineWeb-Edu (HuggingFace): high-quality educational web text
#   - Wikipedia English: world knowledge
#   - OpenHermes-2.5: instruction following + conversations
#   - The Stack Smol: programming knowledge (Python subset)
#   - dolphin-r1: reasoning + general QA
#
# Requirements: pip install datasets tokenizers tqdm
# Total size: ~30-50GB raw, ~8GB tokenized
#
# Mixing ratios:
#   40% FineWeb-Edu + 30% Wikipedia + 15% OpenHermes + 10% Stack + 5% Dolphin

set -euo pipefail

OUTPUT_DIR="${1:-data}"
mkdir -p "$OUTPUT_DIR"

echo "Preparing data in $OUTPUT_DIR ..."

python3 - <<'PYTHON'
import sys, struct, random
from pathlib import Path

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"

try:
    from datasets import load_dataset, interleave_datasets
    from tokenizers import Tokenizer
    from tqdm import tqdm
except ImportError:
    print("ERROR: Install required packages: pip install datasets tokenizers tqdm")
    sys.exit(1)

# Load tokenizer
print("Loading tokenizer...")
try:
    tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
except Exception:
    # Fallback: try local tokenizer.json
    import os
    if os.path.exists("tokenizer.json"):
        tokenizer = Tokenizer.from_file("tokenizer.json")
    else:
        print("ERROR: No tokenizer found. Run: tiny-bit download --model small")
        sys.exit(1)

EOS_ID = tokenizer.token_to_id("</s>") or 2
SEQ_LEN = 1024
TOTAL_TOKENS = 500_000_000  # 500M tokens target

# Dataset configs: (name, config, split, text_field, weight)
datasets_config = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.40),
    ("wikimedia/wikipedia",        "20231101.en", "train", "text", 0.30),
    ("teknium/OpenHermes-2.5",     None,          "train", "conversations", 0.15),
    ("bigcode/the-stack-smol",     "data/python", "train", "content", 0.10),
    ("cognitivecomputations/dolphin-r1", None,    "train", "content", 0.05),
]

def text_from_example(ex, field):
    val = ex.get(field, "")
    if isinstance(val, list):
        return " ".join(str(item.get("value", item)) for item in val)
    return str(val)

print("Loading datasets (streaming)...")
token_buffer = []
train_tokens = []
val_tokens = []

for (ds_name, ds_config, split, field, weight) in datasets_config:
    target = int(TOTAL_TOKENS * weight)
    count = 0
    print(f"  {ds_name}: targeting {target:,} tokens...")
    try:
        if ds_config:
            ds = load_dataset(ds_name, ds_config, split=split, streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)

        for ex in tqdm(ds, desc=ds_name, total=target // 512):
            text = text_from_example(ex, field)
            if not text.strip():
                continue
            enc = tokenizer.encode(text)
            ids = enc.ids + [EOS_ID]
            token_buffer.extend(ids)
            count += len(ids)
            if count >= target:
                break
    except Exception as e:
        print(f"  WARNING: Failed to load {ds_name}: {e}")
        continue

print(f"Total tokens collected: {len(token_buffer):,}")

# Shuffle at document level
random.shuffle(token_buffer)

# Split 98/2 train/val
val_size = max(SEQ_LEN * 100, len(token_buffer) // 50)
val_tokens = token_buffer[:val_size]
train_tokens = token_buffer[val_size:]

def write_bin(tokens, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(tokens)}I", *tokens))
    print(f"  Wrote {len(tokens):,} tokens to {path}")

write_bin(train_tokens, f"{OUTPUT_DIR}/train.bin")
write_bin(val_tokens,   f"{OUTPUT_DIR}/val.bin")
print("Done!")
PYTHON "$OUTPUT_DIR"
