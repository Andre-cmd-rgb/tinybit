#!/bin/bash
# Usage: ./scripts/prepare_data.sh [output_dir]
# Downloads and tokenizes datasets for training tinybit.
#
# Env vars:
#   TOTAL_TOKENS   total desired tokens (default 500_000_000)
#   MIN_TOKENS     fail if fewer than this are collected (default 75% of TOTAL_TOKENS)
#   SEQ_LEN        for val-set sizing (default 1024)
#   DATA_PROFILE   general | coding   (default: general; alias: PROFILE)
#   HF_TOKEN       optional — enables gated datasets (the-stack-smol, etc.)
#   ENABLE_GATED   1 to attempt gated datasets when HF_TOKEN is set (default 1)
#
# Profiles (the ONLY difference between the general and coding model families):
#   general — natural-language assistant mix (FineWeb-Edu, Wikipedia, chat data,
#             a little code).
#   coding  — code-heavy mix (The Stack across several languages, technical
#             chat), with some natural language retained. Code is gated on HF,
#             so set HF_TOKEN for a real coding model.
#
# Prompt-format consistency (IMPORTANT):
#   Conversation/instruction datasets are formatted with the SAME chat template
#   tinybit uses at inference time (see crates/tinybit-core/src/tokenizer.rs):
#       system:
#       {system}
#       user:
#       {user}
#       assistant:
#       {assistant}
#   so the turn structure the model is trained on matches what `tinybit chat`
#   feeds it. Plain corpora (web/wiki/code files) are tokenized as raw text.
#
# Memory design:
#   Tokens are written directly to disk with numpy (4 bytes/token).
#   No in-memory token buffer — peak RAM is proportional to one document at a time.
#   The DataLoader shuffles at training time, so no shuffle is needed here.
#   Val split is taken from the head of the stream (FineWeb-Edu); train is everything after.

set -Eeuo pipefail

OUTPUT_DIR="${1:-data}"
OUTPUT_DIR="${OUTPUT_DIR%/}"  # strip trailing slash to avoid double-slash paths
mkdir -p "$OUTPUT_DIR"

echo "Preparing data in $OUTPUT_DIR ..."

python3 - "$OUTPUT_DIR" <<'PYTHON'
import os, sys
from pathlib import Path

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"

try:
    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tqdm import tqdm
except ImportError:
    print("ERROR: Install required packages: pip install datasets tokenizers tqdm numpy")
    sys.exit(1)

SEQ_LEN       = int(os.environ.get("SEQ_LEN", "1024"))
TOTAL_TOKENS  = int(os.environ.get("TOTAL_TOKENS", "500000000"))
MIN_TOKENS    = int(os.environ.get("MIN_TOKENS", str(int(TOTAL_TOKENS * 0.75))))
HF_TOKEN      = os.environ.get("HF_TOKEN", "").strip() or None
ENABLE_GATED  = os.environ.get("ENABLE_GATED", "1") == "1"
PROFILE       = (os.environ.get("DATA_PROFILE") or os.environ.get("PROFILE") or "general").strip().lower()
if PROFILE not in ("general", "coding"):
    print(f"ERROR: DATA_PROFILE must be 'general' or 'coding' (got {PROFILE!r})")
    sys.exit(1)

# Write buffer: flush to disk every FLUSH_EVERY tokens to keep RAM usage bounded.
# At ~4 bytes/token, 4M tokens = 16 MB peak RAM for the buffer.
FLUSH_EVERY = 4_000_000

print(f"PROFILE={PROFILE}  TOTAL_TOKENS={TOTAL_TOKENS:,}  MIN_TOKENS={MIN_TOKENS:,}  HF_TOKEN={'set' if HF_TOKEN else 'unset'}")

# Canonical chat template — MUST match crates/tinybit-core/src/tokenizer.rs.
SYS_PREFIX  = "system:\n"
USER_PREFIX = "\nuser:\n"
ASST_PREFIX = "\nassistant:\n"

# Tokenizer ------------------------------------------------------------------
print("Loading tokenizer...")
try:
    tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
except Exception:
    if os.path.exists("tokenizer.json"):
        tokenizer = Tokenizer.from_file("tokenizer.json")
    else:
        print("ERROR: No tokenizer found. Run: tinybit download")
        sys.exit(1)

EOS_ID = tokenizer.token_to_id("</s>") or 2

# Dataset tables -------------------------------------------------------------
# (name, config, split, text_field, weight, gated, kind)
#   kind = "text" → tokenize raw text field
#          "chat" → field is a list of role/content turns; format with the
#                   tinybit chat template so training matches inference.
GENERAL_DATASETS = [
    ("HuggingFaceFW/fineweb-edu",         "sample-10BT",  "train", "text",          0.40, False, "text"),
    ("wikimedia/wikipedia",               "20231101.en",  "train", "text",          0.30, False, "text"),
    ("teknium/OpenHermes-2.5",            None,           "train", "conversations", 0.15, False, "chat"),
    ("cognitivecomputations/dolphin-r1",  "nonreasoning", "train", "messages",      0.10, False, "chat"),
    ("bigcode/the-stack-smol",            "data/python",  "train", "content",       0.05, True,  "text"),
]
# Coding mix: code-heavy across several languages + technical chat, with some
# natural language retained so the model still writes coherent prose.
CODING_DATASETS = [
    ("bigcode/the-stack-smol",            "data/python",     "train", "content",       0.22, True,  "text"),
    ("bigcode/the-stack-smol",            "data/rust",       "train", "content",       0.15, True,  "text"),
    ("bigcode/the-stack-smol",            "data/javascript", "train", "content",       0.10, True,  "text"),
    ("bigcode/the-stack-smol",            "data/c",          "train", "content",       0.08, True,  "text"),
    ("bigcode/the-stack-smol",            "data/go",         "train", "content",       0.05, True,  "text"),
    ("teknium/OpenHermes-2.5",            None,              "train", "conversations", 0.20, False, "chat"),
    ("HuggingFaceFW/fineweb-edu",         "sample-10BT",     "train", "text",          0.12, False, "text"),
    ("cognitivecomputations/dolphin-r1",  "nonreasoning",    "train", "messages",      0.08, False, "chat"),
]
DATASETS = CODING_DATASETS if PROFILE == "coding" else GENERAL_DATASETS

if PROFILE == "coding" and not (HF_TOKEN and ENABLE_GATED):
    print("WARNING: coding profile selected but HF_TOKEN is missing/disabled — the gated")
    print("         The Stack code datasets will be skipped and the model will see little")
    print("         to no code. Set HF_TOKEN for a real coding model.")

def normalize_turns(value):
    """Return a list of (role, content) from a conversations/messages field.
    Handles OpenHermes ({'from','value'}) and dolphin/ChatML ({'role','content'})."""
    role_map = {
        "system": "system", "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant",
    }
    turns = []
    if not isinstance(value, list):
        return turns
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("from") or item.get("role") or ""
        content = item.get("value") or item.get("content") or item.get("text") or ""
        role = role_map.get(str(role).lower())
        content = str(content).strip()
        if role and content:
            turns.append((role, content))
    return turns

def format_chat(turns):
    """Render role/content turns with the canonical tinybit chat template."""
    parts = []
    for role, content in turns:
        if role == "system":
            parts.append(f"{SYS_PREFIX}{content}")
        elif role == "user":
            parts.append(f"{USER_PREFIX}{content}")
        elif role == "assistant":
            parts.append(f"{ASST_PREFIX}{content}")
    return "".join(parts)

def text_from_example(ex, field, kind):
    val = ex.get(field, "")
    if kind == "chat":
        return format_chat(normalize_turns(val))
    if isinstance(val, list):
        # Defensive: a "text" field that is unexpectedly a list — join values.
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
    name, cfg, split, field, weight, gated, kind = entry
    if gated and (not HF_TOKEN or not ENABLE_GATED):
        skipped.append((f"{name}:{cfg}", "gated and HF_TOKEN missing/disabled"))
        continue
    active.append(entry)

if skipped:
    print("Skipping datasets:")
    for n, reason in skipped:
        print(f"  - {n}: {reason}")

if not active:
    print("ERROR: no datasets remain active.")
    sys.exit(2)

# Streaming write to disk ----------------------------------------------------
# All tokens are written to a single temp file in stream order.
# We never hold more than FLUSH_EVERY tokens in RAM at once (16 MB peak).
# After collection, the file is split into val + train without loading it whole.

tmp_path = Path(OUTPUT_DIR) / "_tokens_tmp.bin"
collected = {}
total_written = 0
write_buf = []

def flush_buf(f):
    global total_written
    if write_buf:
        arr = np.array(write_buf, dtype=np.uint32)
        arr.tofile(f)
        total_written += len(write_buf)
        write_buf.clear()

remaining_budget = TOTAL_TOKENS
remaining_active = list(active)

with open(tmp_path, 'wb') as tmp_f:
    while remaining_active:
        weight_sum = sum(w for *_, w, _g, _k in remaining_active)
        name, cfg, split, field, weight, _gated, kind = remaining_active.pop(0)
        target = int(remaining_budget * (weight / weight_sum)) if weight_sum > 0 else 0
        if target <= 0:
            continue
        label = f"{name}:{cfg}" if cfg else name
        print(f"  {label}: target {target:,} tokens (kind={kind})")
        got = 0
        try:
            ds = load_stream(name, cfg, split)
            for ex in tqdm(ds, desc=label, unit="ex", mininterval=2.0,
                           total=max(target // 1024, 100)):
                text = text_from_example(ex, field, kind)
                if not text or not text.strip():
                    continue
                try:
                    enc = tokenizer.encode(text)
                except Exception:
                    continue
                ids = enc.ids + [EOS_ID]
                write_buf.extend(ids)
                got += len(ids)
                # Flush buffer periodically to keep RAM bounded
                if len(write_buf) >= FLUSH_EVERY:
                    flush_buf(tmp_f)
                if got >= target:
                    break
            # Flush any remaining tokens for this dataset
            flush_buf(tmp_f)
            collected[label] = got
            remaining_budget = max(0, remaining_budget - got)
            print(f"    {label}: collected {got:,} tokens (remaining budget {remaining_budget:,})")
        except Exception as e:
            print(f"  WARNING: {label} failed: {e}")
            flush_buf(tmp_f)
            collected[label] = 0
            continue

total_collected = total_written
print()
print("Per-dataset collection:")
for n, t in collected.items():
    print(f"  {n}: {t:,}")
print(f"Total collected: {total_collected:,}  (target {TOTAL_TOKENS:,})")

if total_collected < MIN_TOKENS:
    tmp_path.unlink(missing_ok=True)
    print(f"ERROR: only {total_collected:,} tokens collected; MIN_TOKENS={MIN_TOKENS:,}.")
    sys.exit(3)

# Split into val + train without loading entire dataset into RAM -------------
# val = first val_size tokens (head of stream = FineWeb-Edu, highest quality).
# train = everything after val. This ensures val loss reflects the primary
# distribution, not the last/smallest dataset streamed.
val_size = max(SEQ_LEN * 100, total_collected // 50)
val_size = min(val_size, total_collected - SEQ_LEN)  # ensure train has at least seq_len tokens
train_size = total_collected - val_size

COPY_CHUNK = 1 << 20  # copy 4 MB at a time

def copy_range(src_path, dst_path, start_tok, count_tok):
    """Copy [start_tok, start_tok+count_tok) u32 tokens from src to dst."""
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
        src.seek(start_tok * 4)
        remaining = count_tok * 4  # bytes
        while remaining > 0:
            chunk = src.read(min(COPY_CHUNK * 4, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    print(f"  Wrote {count_tok:,} tokens to {dst_path}")

print(f"\nSplitting: val={val_size:,} tokens (head), train={train_size:,} tokens (rest)")
copy_range(tmp_path, f"{OUTPUT_DIR}/val.bin",   0,        val_size)
copy_range(tmp_path, f"{OUTPUT_DIR}/train.bin", val_size, train_size)

tmp_path.unlink(missing_ok=True)
print("Done!")
PYTHON
