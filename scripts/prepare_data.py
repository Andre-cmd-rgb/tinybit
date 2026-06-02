```python
#!/usr/bin/env python3
# tinybit data preparation — robust HuggingFace streaming version.
#
# Fixes the FineWeb-Edu ~60% freeze problem by running each HF streaming dataset
# inside a separate killable process. If the stream hangs, the parent kills the
# child, saves progress, restarts the same dataset, fast-forwards to the last
# processed example, and continues.
#
# Important reality:
# HuggingFace streaming does not support true random-access resume inside a
# remote parquet shard. This script resumes by example offset. That means a
# restart may need to re-scan already-seen examples until it reaches the saved
# offset, but it will not duplicate written tokens and it will not silently skip
# the dataset.
#
# Env vars:
#   TOTAL_TOKENS, MIN_TOKENS, SEQ_LEN, DATA_PROFILE / PROFILE,
#   HF_TOKEN, ENABLE_GATED, CUSTOM_CHAT_DIR, CUSTOM_CHAT_EPOCHS,
#   STREAM_TIMEOUT, MAX_STALLS, MAX_RESTARTS_PER_DATASET,
#   FLUSH_EVERY, RESUME_DATA_PREP

import json
import multiprocessing as mp
import os
import queue as queue_mod
import shutil
import socket
import sys
import time
from pathlib import Path

socket.setdefaulttimeout(60)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tqdm import tqdm
except ImportError:
    print("ERROR: Install required packages: pip install datasets tokenizers tqdm numpy", flush=True)
    sys.exit(1)

SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
TOTAL_TOKENS = int(os.environ.get("TOTAL_TOKENS", "500000000"))
MIN_TOKENS = int(os.environ.get("MIN_TOKENS", str(int(TOTAL_TOKENS * 0.75))))
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
ENABLE_GATED = os.environ.get("ENABLE_GATED", "1") == "1"
PROFILE = (os.environ.get("DATA_PROFILE") or os.environ.get("PROFILE") or "general").strip().lower()

STREAM_TIMEOUT = int(os.environ.get("STREAM_TIMEOUT", "45"))
MAX_STALLS = int(os.environ.get("MAX_STALLS", "2"))
MAX_RESTARTS_PER_DATASET = int(os.environ.get("MAX_RESTARTS_PER_DATASET", "0"))  # 0 = unlimited
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "500000"))
RESUME_DATA_PREP = os.environ.get("RESUME_DATA_PREP", "1") == "1"

CUSTOM_CHAT_DIR = os.environ.get("CUSTOM_CHAT_DIR", "datasets")
CUSTOM_CHAT_EPOCHS = int(os.environ.get("CUSTOM_CHAT_EPOCHS", "10"))

if PROFILE not in ("general", "coding"):
    print(f"ERROR: DATA_PROFILE must be 'general' or 'coding' (got {PROFILE!r})", flush=True)
    sys.exit(1)

print(
    f"PROFILE={PROFILE}  TOTAL_TOKENS={TOTAL_TOKENS:,}  MIN_TOKENS={MIN_TOKENS:,}  "
    f"HF_TOKEN={'set' if HF_TOKEN else 'unset'}",
    flush=True,
)
print(
    f"STREAM_TIMEOUT={STREAM_TIMEOUT}s  MAX_STALLS={MAX_STALLS}  "
    f"MAX_RESTARTS_PER_DATASET={MAX_RESTARTS_PER_DATASET or 'unlimited'}  "
    f"FLUSH_EVERY={FLUSH_EVERY:,}",
    flush=True,
)

SYS_PREFIX = "system:\n"
USER_PREFIX = "\nuser:\n"
ASST_PREFIX = "\nassistant:\n"

PROGRESS_PATH = OUTPUT_DIR / "prepare_progress.json"
TMP_PATH = OUTPUT_DIR / "_tokens_tmp.bin"
TRAIN_PATH = OUTPUT_DIR / "train.bin"
VAL_PATH = OUTPUT_DIR / "val.bin"


GENERAL_DATASETS = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.33, False, "text"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text", 0.30, False, "text"),
    ("roneneldan/TinyStories", None, "train", "text", 0.12, False, "text"),
    ("teknium/OpenHermes-2.5", None, "train", "conversations", 0.15, False, "chat"),
    ("cognitivecomputations/dolphin-r1", "nonreasoning", "train", "messages", 0.07, False, "chat"),
    ("bigcode/the-stack-smol", "data/python", "train", "content", 0.03, True, "text"),
]

CODING_DATASETS = [
    ("bigcode/the-stack-smol", "data/python", "train", "content", 0.22, True, "text"),
    ("bigcode/the-stack-smol", "data/rust", "train", "content", 0.15, True, "text"),
    ("bigcode/the-stack-smol", "data/javascript", "train", "content", 0.10, True, "text"),
    ("bigcode/the-stack-smol", "data/c", "train", "content", 0.08, True, "text"),
    ("bigcode/the-stack-smol", "data/go", "train", "content", 0.05, True, "text"),
    ("teknium/OpenHermes-2.5", None, "train", "conversations", 0.20, False, "chat"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.12, False, "text"),
    ("cognitivecomputations/dolphin-r1", "nonreasoning", "train", "messages", 0.08, False, "chat"),
]


def dataset_label(name, cfg):
    return f"{name}:{cfg}" if cfg else name


def load_tokenizer():
    print("Loading tokenizer...", flush=True)
    try:
        return Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        if Path("tokenizer.json").exists():
            return Tokenizer.from_file("tokenizer.json")
        print("ERROR: No tokenizer found. Run: tinybit download", flush=True)
        sys.exit(1)


tokenizer = load_tokenizer()
EOS_ID = tokenizer.token_to_id("</s>") or 2


def normalize_turns(value):
    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
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


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_stream(name, config, split):
    kwargs = {"split": split, "streaming": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    if config is not None:
        return load_dataset(name, config, **kwargs)
    return load_dataset(name, **kwargs)


def stream_worker(name, config, split, start_example, out_q):
    """
    Child process: owns the HuggingFace iterator.

    If this process wedges inside pyarrow/fsspec/aiohttp, the parent can kill it.
    """
    try:
        ds = load_stream(name, config, split)
        for idx, ex in enumerate(ds):
            if idx < start_example:
                continue
            out_q.put(("ex", idx + 1, ex), block=True)
        out_q.put(("done", None, None), block=True)
    except Exception as exc:
        try:
            out_q.put(("err", None, repr(exc)), block=True)
        except Exception:
            pass


def kill_process(proc):
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


def iter_stream_restartable(name, config, split, label, start_example):
    """
    Yield (example_count_seen, example) and restart from last example count if
    the stream stalls. Does not skip the dataset.
    """
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    cursor = int(start_example)
    restarts = 0

    while True:
        q = ctx.Queue(maxsize=256)
        proc = ctx.Process(
            target=stream_worker,
            args=(name, config, split, cursor, q),
            daemon=True,
        )
        proc.start()

        print(f"  [stream] {label}: start worker pid={proc.pid} from example offset {cursor:,}", flush=True)

        stalls = 0
        made_progress = False

        try:
            while True:
                try:
                    kind, seen, payload = q.get(timeout=STREAM_TIMEOUT)
                except queue_mod.Empty:
                    stalls += 1
                    print(
                        f"  [stall] {label}: no example for {STREAM_TIMEOUT}s "
                        f"(strike {stalls}/{MAX_STALLS}) at example offset {cursor:,}",
                        flush=True,
                    )

                    if stalls >= MAX_STALLS:
                        kill_process(proc)
                        restarts += 1

                        if MAX_RESTARTS_PER_DATASET and restarts > MAX_RESTARTS_PER_DATASET:
                            raise RuntimeError(
                                f"{label} stalled too many times at example offset {cursor:,}; "
                                f"raise MAX_RESTARTS_PER_DATASET or use a different data source."
                            )

                        backoff = min(60, 5 * restarts)
                        print(
                            f"  [restart] {label}: restarting same dataset from "
                            f"example offset {cursor:,} after {backoff}s "
                            f"(restart #{restarts})",
                            flush=True,
                        )
                        time.sleep(backoff)
                        break

                    continue

                stalls = 0

                if kind == "ex":
                    cursor = int(seen)
                    made_progress = True
                    yield cursor, payload
                elif kind == "done":
                    kill_process(proc)
                    print(f"  [done] {label}: stream finished at example offset {cursor:,}", flush=True)
                    return
                elif kind == "err":
                    kill_process(proc)
                    restarts += 1
                    print(
                        f"  [warn] {label}: stream error {payload}; restarting "
                        f"from example offset {cursor:,} (restart #{restarts})",
                        flush=True,
                    )
                    time.sleep(min(60, 5 * restarts))
                    break

                if not proc.is_alive() and q.empty():
                    return

        finally:
            kill_process(proc)

        if not made_progress:
            print(
                f"  [note] {label}: restart made no progress; still retrying same dataset, "
                f"not skipping it.",
                flush=True,
            )


def get_custom_files():
    files = []
    if CUSTOM_CHAT_DIR and Path(CUSTOM_CHAT_DIR).is_dir() and CUSTOM_CHAT_EPOCHS > 0:
        for fn in sorted(os.listdir(CUSTOM_CHAT_DIR)):
            if fn.endswith((".jsonl", ".txt")):
                files.append(str(Path(CUSTOM_CHAT_DIR) / fn))
    if files:
        print(
            f"Custom chat data: {len(files)} file(s) from {CUSTOM_CHAT_DIR}/ "
            f"x{CUSTOM_CHAT_EPOCHS} epochs (tokenized last): "
            + ", ".join(Path(f).name for f in files),
            flush=True,
        )
    return files


custom_files = get_custom_files()


def active_datasets():
    table = CODING_DATASETS if PROFILE == "coding" else GENERAL_DATASETS
    active = []
    skipped = []

    for entry in table:
        name, cfg, split, field, weight, gated, kind = entry
        if gated and (not HF_TOKEN or not ENABLE_GATED):
            skipped.append((dataset_label(name, cfg), "gated and HF_TOKEN missing/disabled"))
            continue
        active.append(entry)

    if skipped:
        print("Skipping datasets:", flush=True)
        for n, reason in skipped:
            print(f"  - {n}: {reason}", flush=True)

    if not active:
        print("ERROR: no datasets remain active.", flush=True)
        sys.exit(2)

    return active


DATASETS = active_datasets()


def fresh_progress():
    return {
        "version": 2,
        "profile": PROFILE,
        "total_tokens_target": TOTAL_TOKENS,
        "min_tokens": MIN_TOKENS,
        "total_written": 0,
        "datasets": {},
        "current_index": 0,
        "custom_done": False,
    }


def load_progress():
    if RESUME_DATA_PREP and PROGRESS_PATH.exists() and TMP_PATH.exists():
        try:
            with open(PROGRESS_PATH, encoding="utf-8") as fh:
                p = json.load(fh)
            if p.get("version") == 2 and p.get("profile") == PROFILE:
                expected_bytes = int(p.get("total_written", 0)) * 4
                actual_bytes = TMP_PATH.stat().st_size
                if actual_bytes >= expected_bytes:
                    if actual_bytes != expected_bytes:
                        print(
                            f"[resume] truncating temp token file from {actual_bytes} to "
                            f"{expected_bytes} bytes to match progress",
                            flush=True,
                        )
                        with open(TMP_PATH, "r+b") as fh2:
                            fh2.truncate(expected_bytes)
                    print(
                        f"[resume] loaded progress: {p.get('total_written', 0):,} tokens written",
                        flush=True,
                    )
                    return p
        except Exception as exc:
            print(f"[resume] ignored broken progress file: {exc!r}", flush=True)

    if not RESUME_DATA_PREP:
        print("[resume] RESUME_DATA_PREP=0 — starting clean", flush=True)
    return fresh_progress()


progress = load_progress()


def save_progress():
    tmp = PROGRESS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, indent=2, sort_keys=True)
    os.replace(tmp, PROGRESS_PATH)


write_buf = []


def flush_buf(fh):
    if not write_buf:
        return
    arr = np.array(write_buf, dtype=np.uint32)
    arr.tofile(fh)
    progress["total_written"] = int(progress.get("total_written", 0)) + len(write_buf)
    write_buf.clear()
    fh.flush()
    os.fsync(fh.fileno())
    save_progress()


def tokenize_to_buffer(text):
    if not text or not text.strip():
        return 0
    try:
        enc = tokenizer.encode(text)
    except Exception:
        return 0
    ids = enc.ids + [EOS_ID]
    write_buf.extend(ids)
    return len(ids)


def dataset_target(remaining_budget, remaining_active):
    weight_sum = sum(w for *_, w, _g, _k in remaining_active)
    if weight_sum <= 0:
        return 0
    _name, _cfg, _split, _field, weight, _gated, _kind = remaining_active[0]
    return int(remaining_budget * (weight / weight_sum))


def prepare_weighted_mix():
    remaining_budget = max(0, TOTAL_TOKENS - int(progress.get("total_written", 0)))

    mode = "ab" if TMP_PATH.exists() else "wb"
    with open(TMP_PATH, mode) as tmp_f:
        start_index = int(progress.get("current_index", 0))

        while start_index < len(DATASETS):
            remaining_active = DATASETS[start_index:]
            name, cfg, split, field, weight, _gated, kind = remaining_active[0]
            label = dataset_label(name, cfg)

            ds_state = progress["datasets"].setdefault(
                label,
                {
                    "done": False,
                    "examples_seen": 0,
                    "tokens": 0,
                    "target": None,
                    "restarts": 0,
                },
            )

            if ds_state.get("done"):
                print(f"  {label}: already done — skipping to next dataset", flush=True)
                start_index += 1
                progress["current_index"] = start_index
                save_progress()
                continue

            target = ds_state.get("target")
            if target is None:
                target = dataset_target(remaining_budget, remaining_active)
                ds_state["target"] = int(target)
                save_progress()

            if target <= 0:
                ds_state["done"] = True
                start_index += 1
                progress["current_index"] = start_index
                save_progress()
                continue

            print(
                f"  {label}: target {target:,} tokens (kind={kind}) "
                f"resume_examples={int(ds_state.get('examples_seen', 0)):,} "
                f"resume_tokens={int(ds_state.get('tokens', 0)):,}",
                flush=True,
            )

            got = int(ds_state.get("tokens", 0))
            start_example = int(ds_state.get("examples_seen", 0))

            pbar_total = max(target // 1024, 100)
            pbar = tqdm(
                desc=label,
                unit="ex",
                mininterval=5.0,
                total=pbar_total,
                initial=max(0, min(start_example, pbar_total)),
            )

            try:
                for examples_seen, ex in iter_stream_restartable(name, cfg, split, label, start_example):
                    if got >= target:
                        break

                    text = text_from_example(ex, field, kind)
                    added = tokenize_to_buffer(text)
                    got += added

                    ds_state["examples_seen"] = int(examples_seen)
                    ds_state["tokens"] = int(got)

                    pbar.update(1)

                    if len(write_buf) >= FLUSH_EVERY:
                        flush_buf(tmp_f)

                flush_buf(tmp_f)
                pbar.close()

                ds_state["tokens"] = int(got)
                ds_state["done"] = True
                remaining_budget = max(0, remaining_budget - got)

                print(
                    f"    {label}: collected {got:,} tokens "
                    f"(remaining budget {remaining_budget:,})",
                    flush=True,
                )

                start_index += 1
                progress["current_index"] = start_index
                save_progress()

            except Exception as exc:
                pbar.close()
                flush_buf(tmp_f)
                save_progress()
                print(f"ERROR: {label} failed without being skipped: {exc}", flush=True)
                raise


def prepare_custom_chat():
    if progress.get("custom_done"):
        print("Custom chat data already done — skipping", flush=True)
        return

    if not custom_files:
        progress["custom_done"] = True
        save_progress()
        return

    with open(TMP_PATH, "ab") as tmp_f:
        for fp in custom_files:
            label = f"custom:{Path(fp).name}"
            state = progress["datasets"].setdefault(
                label,
                {"done": False, "tokens": 0, "examples_seen": 0},
            )

            if state.get("done"):
                continue

            got = int(state.get("tokens", 0))
            seen = int(state.get("examples_seen", 0))
            global_seen = 0

            print(f"  {label}: tokenizing custom chat x{CUSTOM_CHAT_EPOCHS}", flush=True)

            for _epoch in range(CUSTOM_CHAT_EPOCHS):
                for obj in read_jsonl(fp):
                    global_seen += 1
                    if global_seen <= seen:
                        continue

                    turns = normalize_turns(obj.get("messages", []))
                    roles = {r for r, _ in turns}
                    if "user" not in roles or "assistant" not in roles:
                        continue

                    text = format_chat(turns)
                    added = tokenize_to_buffer(text)
                    got += added
                    state["tokens"] = int(got)
                    state["examples_seen"] = int(global_seen)

                    if len(write_buf) >= FLUSH_EVERY:
                        flush_buf(tmp_f)
                        save_progress()

            flush_buf(tmp_f)
            state["done"] = True
            save_progress()
            print(f"  {label}: {got:,} tokens", flush=True)

    progress["custom_done"] = True
    save_progress()


def copy_range(src_path, dst_path, start_tok, count_tok):
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    copy_chunk_tokens = 1 << 20

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        src.seek(start_tok * 4)
        remaining = count_tok * 4
        while remaining > 0:
            chunk = src.read(min(copy_chunk_tokens * 4, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)

    print(f"  Wrote {count_tok:,} tokens to {dst_path}", flush=True)


def split_train_val():
    total_collected = int(progress.get("total_written", 0))

    print("\nPer-dataset collection:", flush=True)
    for label, state in progress.get("datasets", {}).items():
        print(f"  {label}: {int(state.get('tokens', 0)):,}", flush=True)

    print(f"Total collected: {total_collected:,}  (target {TOTAL_TOKENS:,})", flush=True)

    if total_collected < MIN_TOKENS:
        print(
            f"ERROR: only {total_collected:,} tokens collected; "
            f"MIN_TOKENS={MIN_TOKENS:,}. Keeping {TMP_PATH} and {PROGRESS_PATH} "
            f"so the next run can resume.",
            flush=True,
        )
        sys.exit(3)

    val_size = max(SEQ_LEN * 100, total_collected // 50)
    val_size = min(val_size, total_collected - SEQ_LEN)
    train_size = total_collected - val_size

    print(
        f"\nSplitting: val={val_size:,} tokens (head), "
        f"train={train_size:,} tokens (rest)",
        flush=True,
    )

    copy_range(TMP_PATH, VAL_PATH, 0, val_size)
    copy_range(TMP_PATH, TRAIN_PATH, val_size, train_size)

    try:
        TMP_PATH.unlink()
    except FileNotFoundError:
        pass

    progress["finalized"] = True
    save_progress()
    print("Done!", flush=True)


def main():
    if TRAIN_PATH.exists() and VAL_PATH.exists() and TRAIN_PATH.stat().st_size > 0 and VAL_PATH.stat().st_size > 0:
        print("data/train.bin and data/val.bin already exist — nothing to do.", flush=True)
        return

    prepare_weighted_mix()
    prepare_custom_chat()
    split_train_val()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Progress kept; rerun to resume.", flush=True)
        sys.exit(130)
```
