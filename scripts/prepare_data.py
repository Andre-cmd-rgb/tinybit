#!/usr/bin/env python3
"""tinybit data preparation — robust, resumable HuggingFace streaming.

Root cause this guards against
------------------------------
HuggingFace streaming (fineweb-edu sample-10BT, ~62% in) deterministically
wedged inside a C-level pyarrow/fsspec/aiohttp read. None of the earlier
single-process mitigations could recover from it:

  * ``socket.setdefaulttimeout`` does not cover HF's streaming read path.
  * a SIGALRM watchdog cannot interrupt a thread blocked in a C call (the
    Python signal handler only runs when the interpreter regains control).
  * a producer *thread* + bounded queue lets the consumer time out, but the
    wedged thread cannot be killed — it leaks a CLOSE-WAIT socket and the run
    freezes at the same shard boundary forever.

The only reliable fix is to own the HF iterator in a separate *process* that
the parent can ``terminate()``/``kill()``. This module does exactly that and
adds the operational guarantees a multi-hour, unattended cloud run needs:

  * **Resumability / checkpointing** — the worker periodically ships HF's
    ``IterableDataset.state_dict()`` back to the parent, which persists it.
    A restart reloads that state and resumes near the wedge point instead of
    re-scanning from example 0. If a dataset does not support stateful
    streaming, it transparently falls back to resume-by-example-offset.
  * **Timeouts** — socket + HF download/etag timeouts, plus a per-example
    queue ``get`` timeout that always returns regardless of what the child does.
  * **Retries / backoff** — a wedged or erroring child is killed and the
    dataset is restarted with exponential backoff.
  * **Bounded failure** — if a dataset makes *zero* progress across
    ``MAX_RESTARTS_NO_PROGRESS`` consecutive restarts it FAILS LOUDLY (writes
    a failure marker, exits non-zero) instead of looping forever.
  * **Progress persistence** — token output and per-dataset cursors are
    fsync'd to ``prepare_progress.json`` + ``_tokens_tmp.bin`` so any restart
    (preemption, reboot, crash) resumes without re-tokenizing.
  * **Heartbeat + structured logs** — every event is emitted as one JSON line
    and a fresh ``heartbeat_unix`` is written to the progress file, so the
    surrounding orchestration (startup.sh) can sync visibility to the bucket
    and a watchdog can distinguish "alive and working" from "wedged".

Env vars
--------
  TOTAL_TOKENS, MIN_TOKENS, SEQ_LEN, DATA_PROFILE / PROFILE,
  HF_TOKEN, ENABLE_GATED, CUSTOM_CHAT_DIR, CUSTOM_CHAT_EPOCHS,
  STREAM_TIMEOUT, MAX_STALLS, MAX_RESTARTS_PER_DATASET,
  MAX_RESTARTS_NO_PROGRESS, CHECKPOINT_EVERY, FLUSH_EVERY, RESUME_DATA_PREP

Test seam (no network)
----------------------
  TINYBIT_FAKE_STREAM=1 swaps the HF loader + tokenizer for deterministic
  in-memory fakes so the kill/restart/resume machinery can be exercised
  offline. See scripts/test_prepare_data.py.
"""

import base64
import json
import multiprocessing as mp
import os
import pickle
import queue as queue_mod
import socket
import sys
import time
from pathlib import Path

# Network hardening. These help *normal* failures fail fast; the killable-child
# architecture below is what handles the C-level wedge they cannot cover.
socket.setdefaulttimeout(60)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")

# ---------------------------------------------------------------------------
# Configuration (cheap, no I/O — safe to evaluate at import on every spawn).
# ---------------------------------------------------------------------------
FAKE_STREAM = os.environ.get("TINYBIT_FAKE_STREAM", "") not in ("", "0")

SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
TOTAL_TOKENS = int(os.environ.get("TOTAL_TOKENS", "500000000"))
MIN_TOKENS = int(os.environ.get("MIN_TOKENS", str(int(TOTAL_TOKENS * 0.75))))
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
ENABLE_GATED = os.environ.get("ENABLE_GATED", "1") == "1"
PROFILE = (os.environ.get("DATA_PROFILE") or os.environ.get("PROFILE") or "general").strip().lower()

STREAM_TIMEOUT = int(os.environ.get("STREAM_TIMEOUT", "45"))
MAX_STALLS = int(os.environ.get("MAX_STALLS", "2"))
# Hard cap on TOTAL restarts for one dataset (0 = unlimited). The meaningful
# guard is MAX_RESTARTS_NO_PROGRESS below; this is just a final backstop.
MAX_RESTARTS_PER_DATASET = int(os.environ.get("MAX_RESTARTS_PER_DATASET", "0"))
# Consecutive restarts that yield ZERO new examples before we give up and FAIL.
# This is what kills a deterministic dead-shard stall instead of looping forever.
MAX_RESTARTS_NO_PROGRESS = int(os.environ.get("MAX_RESTARTS_NO_PROGRESS", "8"))
# How often (in examples) the worker ships an IterableDataset.state_dict() back
# to the parent for cheap resume. 0 disables stateful checkpointing.
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "5000"))
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "500000"))
RESUME_DATA_PREP = os.environ.get("RESUME_DATA_PREP", "1") == "1"

CUSTOM_CHAT_DIR = os.environ.get("CUSTOM_CHAT_DIR", "datasets")
CUSTOM_CHAT_EPOCHS = int(os.environ.get("CUSTOM_CHAT_EPOCHS", "10"))

SYS_PREFIX = "system:\n"
USER_PREFIX = "\nuser:\n"
ASST_PREFIX = "\nassistant:\n"

# General mix, retuned 2026-06-07 for a language-first ~50M assistant:
# English fluency + comprehension + summarising + instruction-following, NO code,
# and facts handled by the `lookup` tool rather than memorised (a tiny model
# can't store facts reliably — see the lookup tool / datasets/chat-lookup-05).
#   - the-stack (Python code) REMOVED — this build is not a coding model.
#   - OpenHermes + dolphin bumped (instruction following, summarising, Q&A,
#     rewriting — the skills we actually want).
#   - TinyStories bumped (clean, simple English — punches above its weight at 50M).
#   - FineWeb-Edu stays the English backbone and the val-set head.
#   - Cosmopedia kept but trimmed: great for coherent English/reasoning, but it's
#     textbook-dense, and we're deliberately not optimising for fact recall.
GENERAL_DATASETS = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.32, False, "text"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text", 0.25, False, "text"),
    ("roneneldan/TinyStories", None, "train", "text", 0.15, False, "text"),
    ("teknium/OpenHermes-2.5", None, "train", "conversations", 0.20, False, "chat"),
    ("cognitivecomputations/dolphin-r1", "nonreasoning", "train", "messages", 0.08, False, "chat"),
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


# ---------------------------------------------------------------------------
# Structured logging.
# ---------------------------------------------------------------------------
def now_unix():
    return int(time.time())


def log_event(event, **fields):
    """Emit one structured JSON line (machine-parseable) + flush immediately."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event}
    rec.update(fields)
    print("EVENT " + json.dumps(rec, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (text/chat normalization) — no I/O, import-safe.
# ---------------------------------------------------------------------------
def dataset_label(name, cfg):
    return f"{name}:{cfg}" if cfg else name


def is_permanent_error(msg):
    """True for errors that will NEVER succeed on retry (gating, auth, missing
    dataset) — so we skip the source immediately instead of burning the restart
    budget. Transient network/stream errors are NOT matched and still retry."""
    if not msg:
        return False
    m = str(msg).lower()
    needles = (
        "gated dataset", "ask for access", "datasetnotfounderror", "gatedrepoerror",
        "repositorynotfounderror", "doesn't exist", "does not exist", "not found",
        "401", "403", "404", "unauthorized", "forbidden", "authentication",
    )
    return any(n in m for n in needles)


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


# ---------------------------------------------------------------------------
# HF stream loading + the killable worker process.
# These live at module scope so they are importable and picklable for the
# "spawn" start method (Windows / macOS); on Linux "fork" is used.
# ---------------------------------------------------------------------------
def _fake_stream():
    """Deterministic in-memory stand-in for an HF IterableDataset (test seam).

    Driven by env vars so a spawned child reconstructs the same behavior:
      TINYBIT_FAKE_N       total examples (default 30000)
      TINYBIT_FAKE_HANG_AT example index at which the FIRST incarnation wedges
      TINYBIT_FAKE_HANG_FLAG  file path used to make the wedge happen only once
                              (first run creates it then hangs; a restart sees
                              it and runs clean, simulating a transient stall)
    """
    n = int(os.environ.get("TINYBIT_FAKE_N", "30000"))
    hang_at = int(os.environ.get("TINYBIT_FAKE_HANG_AT", "-1"))
    flag = os.environ.get("TINYBIT_FAKE_HANG_FLAG", "")

    class _Fake:
        def __init__(self):
            self._pos = 0

        def state_dict(self):
            return {"pos": self._pos}

        def load_state_dict(self, state):
            self._pos = int(state.get("pos", 0))

        def __iter__(self):
            i = self._pos
            while i < n:
                if hang_at >= 0 and i == hang_at:
                    do_hang = True
                    if flag:
                        if os.path.exists(flag):
                            do_hang = False
                        else:
                            try:
                                Path(flag).write_text("hung")
                            except Exception:
                                pass
                    if do_hang:
                        while True:  # simulate an un-interruptible C-level wedge
                            time.sleep(3600)
                self._pos = i + 1
                yield {"text": f"fake example number {i} " + ("lorem ipsum " * 8)}
                i += 1

    return _Fake()


def load_stream(name, config, split):
    if FAKE_STREAM:
        return _fake_stream()
    from datasets import load_dataset  # imported lazily so the test seam needs no deps

    kwargs = {"split": split, "streaming": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    if config is not None:
        return load_dataset(name, config, **kwargs)
    return load_dataset(name, **kwargs)


def _try_load_state(ds, state_b64):
    """Best-effort: position `ds` at a saved state_dict. Returns True on success."""
    if not state_b64 or not hasattr(ds, "load_state_dict"):
        return False
    try:
        state = pickle.loads(base64.b64decode(state_b64.encode("ascii")))
        ds.load_state_dict(state)
        return True
    except Exception as exc:  # noqa: BLE001 — any failure falls back to offset re-scan
        log_event("resume_state_failed", error=repr(exc))
        return False


def stream_worker(name, config, split, resume_state_b64, resume_base, start_example, ckpt_every, out_q):
    """Child process: owns the HF iterator and ships examples to the parent.

    Messages put on the queue (always blocking, so backpressure is honored):
      ("ckpt", global_idx, state_b64)  periodic resume checkpoint
      ("scan", global_idx, None)       progress heartbeat while skipping
      ("ex",   global_idx+1, example)  a real example to tokenize
      ("done", None, None)             stream exhausted
      ("err",  None, repr(exc))        the iterator raised
    """
    try:
        ds = load_stream(name, config, split)

        base = 0
        if _try_load_state(ds, resume_state_b64):
            base = int(resume_base)
            out_q.put(("scan", base, None), block=True)
        elif start_example > 0:
            # No usable checkpoint: we must re-scan from 0 to start_example.
            out_q.put(("rescan", start_example, None), block=True)

        for local_idx, ex in enumerate(ds):
            global_idx = base + local_idx

            if global_idx < start_example:
                if (global_idx % 5000) == 0:
                    out_q.put(("scan", global_idx, None), block=True)
                continue

            if ckpt_every and (global_idx % ckpt_every == 0) and hasattr(ds, "state_dict"):
                try:
                    # state_dict() here reflects "consumed through global_idx", so a
                    # reload resumes at global_idx + 1 — that is the resume base.
                    st = base64.b64encode(pickle.dumps(ds.state_dict())).decode("ascii")
                    out_q.put(("ckpt", global_idx + 1, st), block=True)
                except Exception:  # noqa: BLE001 — checkpointing is best-effort
                    pass

            out_q.put(("ex", global_idx + 1, ex), block=True)

        out_q.put(("done", None, None), block=True)
    except Exception as exc:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# DataPrep — owns progress, the temp token file, and the restartable streams.
# ---------------------------------------------------------------------------
class DataPrep:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.PROGRESS_PATH = self.output_dir / "prepare_progress.json"
        self.TMP_PATH = self.output_dir / "_tokens_tmp.bin"
        self.TRAIN_PATH = self.output_dir / "train.bin"
        self.VAL_PATH = self.output_dir / "val.bin"
        self.FAILED_PATH = self.output_dir / "prepare_FAILED.json"

        self.mp_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

        import numpy as np  # local import keeps the test seam dependency-light

        self.np = np
        self.write_buf = []
        self._last_hb = 0
        # Cheap setup only. Load progress and write a heartbeat IMMEDIATELY so the
        # orchestration watchdog sees liveness before the (network-bound, possibly
        # slow) tokenizer load — a hang there then trips the heartbeat check.
        self.progress = self.load_progress()
        self.save_progress(phase="starting", last_event="init")
        self.tokenizer = None
        self.eos_id = 2
        self.custom_files = []
        self.datasets = []

    def _setup(self):
        """Heavy initialization (tokenizer + dataset selection). Deferred out of
        __init__ so the heartbeat file exists first."""
        self.tokenizer = self.load_tokenizer()
        self.eos_id = self.tokenizer.token_to_id("</s>") or 2
        self.custom_files = self.get_custom_files()
        self.datasets = self.active_datasets()

    # ---- tokenizer -------------------------------------------------------
    def load_tokenizer(self):
        if FAKE_STREAM:
            class _FakeTok:
                class _Enc:
                    def __init__(self, ids):
                        self.ids = ids

                def encode(self, text):
                    return _FakeTok._Enc([1] * (len(text.split()) + 1))

                def token_to_id(self, _):
                    return 2

            return _FakeTok()

        from tokenizers import Tokenizer

        log_event("tokenizer_loading")
        try:
            tok = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        except Exception:
            if Path("tokenizer.json").exists():
                tok = Tokenizer.from_file("tokenizer.json")
            else:
                log_event("fatal", reason="no tokenizer; run: tinybit download")
                sys.exit(1)
        log_event("tokenizer_loaded")
        return tok

    # ---- dataset selection ----------------------------------------------
    def get_custom_files(self):
        files = []
        if CUSTOM_CHAT_DIR and Path(CUSTOM_CHAT_DIR).is_dir() and CUSTOM_CHAT_EPOCHS > 0:
            for fn in sorted(os.listdir(CUSTOM_CHAT_DIR)):
                if fn.endswith((".jsonl", ".txt")):
                    files.append(str(Path(CUSTOM_CHAT_DIR) / fn))
        if files:
            log_event("custom_chat_found", count=len(files), epochs=CUSTOM_CHAT_EPOCHS,
                      files=[Path(f).name for f in files])
        return files

    def active_datasets(self):
        if FAKE_STREAM:
            # A single fake dataset is enough to exercise the machinery offline.
            return [("fake/dataset", None, "train", "text", 1.0, False, "text")]

        table = CODING_DATASETS if PROFILE == "coding" else GENERAL_DATASETS
        active, skipped = [], []
        for entry in table:
            name, cfg, _split, _field, _weight, gated, _kind = entry
            if gated and (not HF_TOKEN or not ENABLE_GATED):
                skipped.append(dataset_label(name, cfg))
                continue
            active.append(entry)
        if skipped:
            log_event("datasets_skipped", reason="gated and HF_TOKEN missing/disabled", datasets=skipped)
        if not active:
            log_event("fatal", reason="no datasets remain active")
            sys.exit(2)
        log_event("datasets_active", count=len(active),
                  datasets=[dataset_label(e[0], e[1]) for e in active])
        return active

    # ---- progress persistence -------------------------------------------
    def fresh_progress(self):
        return {
            "version": 3,
            "profile": PROFILE,
            "total_tokens_target": TOTAL_TOKENS,
            "min_tokens": MIN_TOKENS,
            "total_written": 0,
            "datasets": {},
            "current_index": 0,
            "custom_done": False,
            "heartbeat_unix": now_unix(),
            "phase": "init",
            "last_event": "fresh",
        }

    def load_progress(self):
        if RESUME_DATA_PREP and self.PROGRESS_PATH.exists() and self.TMP_PATH.exists():
            try:
                with open(self.PROGRESS_PATH, encoding="utf-8") as fh:
                    p = json.load(fh)
                if p.get("version") == 3 and p.get("profile") == PROFILE:
                    expected_bytes = int(p.get("total_written", 0)) * 4
                    actual_bytes = self.TMP_PATH.stat().st_size
                    if actual_bytes >= expected_bytes:
                        if actual_bytes != expected_bytes:
                            log_event("resume_truncate", from_bytes=actual_bytes, to_bytes=expected_bytes)
                            with open(self.TMP_PATH, "r+b") as fh2:
                                fh2.truncate(expected_bytes)
                        log_event("resume_loaded", total_written=int(p.get("total_written", 0)))
                        return p
                    log_event("resume_rejected", reason="temp file shorter than progress",
                              actual_bytes=actual_bytes, expected_bytes=expected_bytes)
            except Exception as exc:  # noqa: BLE001
                log_event("resume_broken", error=repr(exc))
        if not RESUME_DATA_PREP:
            log_event("resume_disabled")
        return self.fresh_progress()

    def save_progress(self, phase=None, last_event=None):
        if phase is not None:
            self.progress["phase"] = phase
        if last_event is not None:
            self.progress["last_event"] = last_event
        self.progress["heartbeat_unix"] = now_unix()
        tmp = self.PROGRESS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.progress, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.PROGRESS_PATH)

    def heartbeat(self, phase=None, last_event=None):
        """Cheap liveness update — does NOT rewrite the whole file every call."""
        # Rewriting the JSON a few times per second would be wasteful; throttle.
        last = getattr(self, "_last_hb", 0)
        t = now_unix()
        if t - last >= 5 or phase is not None or last_event is not None:
            self._last_hb = t
            self.save_progress(phase=phase, last_event=last_event)

    # ---- token buffer ----------------------------------------------------
    def flush_buf(self, fh):
        if not self.write_buf:
            return
        arr = self.np.array(self.write_buf, dtype=self.np.uint32)
        arr.tofile(fh)
        self.progress["total_written"] = int(self.progress.get("total_written", 0)) + len(self.write_buf)
        self.write_buf.clear()
        fh.flush()
        os.fsync(fh.fileno())
        self.save_progress()

    def tokenize_to_buffer(self, text):
        if not text or not text.strip():
            return 0
        try:
            enc = self.tokenizer.encode(text)
        except Exception:
            return 0
        ids = enc.ids + [self.eos_id]
        self.write_buf.extend(ids)
        return len(ids)

    # ---- restartable stream ---------------------------------------------
    def iter_stream_restartable(self, name, config, split, label, ds_state):
        """Yield (examples_seen, example), restarting the killable child on stall.

        Resume strategy, in order of preference:
          1. HF state_dict (cheap): reload exact shard/row position.
          2. example-offset re-scan (always correct, slower).
        Fails loudly if the dataset makes zero progress across
        MAX_RESTARTS_NO_PROGRESS consecutive restarts.
        """
        cursor = int(ds_state.get("examples_seen", 0))
        state_b64 = ds_state.get("hf_state_b64") or ""
        state_base = int(ds_state.get("hf_state_base", 0))
        restarts = 0
        restarts_no_progress = 0

        while True:
            q = self.mp_ctx.Queue(maxsize=256)
            proc = self.mp_ctx.Process(
                target=stream_worker,
                args=(name, config, split, state_b64, state_base, cursor, CHECKPOINT_EVERY, q),
                daemon=True,
            )
            proc.start()
            log_event("stream_start", dataset=label, pid=proc.pid, cursor=cursor,
                      has_checkpoint=bool(state_b64))

            stalls = 0
            progressed_this_run = False
            try:
                while True:
                    try:
                        kind, seen, payload = q.get(timeout=STREAM_TIMEOUT)
                    except queue_mod.Empty:
                        stalls += 1
                        self.heartbeat(last_event=f"stall:{label}:{stalls}")
                        log_event("stall", dataset=label, strike=stalls, max=MAX_STALLS, cursor=cursor)
                        if stalls >= MAX_STALLS:
                            kill_process(proc)
                            restarts += 1
                            if not progressed_this_run:
                                restarts_no_progress += 1
                            else:
                                restarts_no_progress = 0
                            self._check_restart_budget(label, restarts, restarts_no_progress, cursor)
                            backoff = min(60, 5 * restarts)
                            log_event("restart", dataset=label, restart=restarts,
                                      no_progress=restarts_no_progress, backoff_s=backoff, cursor=cursor)
                            self.heartbeat(last_event=f"restart:{label}:{restarts}")
                            time.sleep(backoff)
                            break
                        continue

                    stalls = 0

                    if kind == "ex":
                        cursor = int(seen)
                        progressed_this_run = True
                        restarts_no_progress = 0
                        yield cursor, payload
                    elif kind == "ckpt":
                        state_b64 = payload
                        state_base = int(seen)
                        ds_state["hf_state_b64"] = state_b64
                        ds_state["hf_state_base"] = state_base
                        self.heartbeat()
                    elif kind in ("scan", "rescan"):
                        if kind == "rescan":
                            log_event("rescan", dataset=label, to_example=int(seen),
                                      note="no usable checkpoint; re-scanning from 0")
                        self.heartbeat(last_event=f"scan:{label}:{seen}")
                    elif kind == "done":
                        kill_process(proc)
                        log_event("stream_done", dataset=label, cursor=cursor)
                        return
                    elif kind == "err":
                        kill_process(proc)
                        if is_permanent_error(payload):
                            # Gating/auth/missing — retrying can never succeed. Give
                            # up on THIS dataset now; the caller decides whether the
                            # run can still finalize (MIN_TOKENS) or must fail.
                            log_event("stream_permanent_error", dataset=label, error=payload, cursor=cursor)
                            raise RuntimeError(f"{label}: permanent/unavailable source, not retrying: {payload}")
                        restarts += 1
                        if not progressed_this_run:
                            restarts_no_progress += 1
                        else:
                            restarts_no_progress = 0
                        log_event("stream_error", dataset=label, error=payload,
                                  restart=restarts, no_progress=restarts_no_progress, cursor=cursor)
                        self._check_restart_budget(label, restarts, restarts_no_progress, cursor)
                        self.heartbeat(last_event=f"error_restart:{label}:{restarts}")
                        time.sleep(min(60, 5 * restarts))
                        break

                    if not proc.is_alive() and q.empty():
                        return
            finally:
                kill_process(proc)

    def _check_restart_budget(self, label, restarts, restarts_no_progress, cursor):
        if restarts_no_progress >= MAX_RESTARTS_NO_PROGRESS:
            raise RuntimeError(
                f"{label}: {restarts_no_progress} consecutive restarts made no progress "
                f"at example offset {cursor:,}; the source appears permanently stalled. "
                f"Raise MAX_RESTARTS_NO_PROGRESS or change the data source."
            )
        if MAX_RESTARTS_PER_DATASET and restarts > MAX_RESTARTS_PER_DATASET:
            raise RuntimeError(
                f"{label}: exceeded MAX_RESTARTS_PER_DATASET={MAX_RESTARTS_PER_DATASET} "
                f"at example offset {cursor:,}."
            )

    # ---- weighted mix ----------------------------------------------------
    def dataset_target(self, remaining_budget, remaining_active):
        weight_sum = sum(w for *_, w, _g, _k in remaining_active)
        if weight_sum <= 0:
            return 0
        weight = remaining_active[0][4]
        return int(remaining_budget * (weight / weight_sum))

    def prepare_weighted_mix(self):
        remaining_budget = max(0, TOTAL_TOKENS - int(self.progress.get("total_written", 0)))
        mode = "ab" if self.TMP_PATH.exists() else "wb"
        with open(self.TMP_PATH, mode) as tmp_f:
            start_index = int(self.progress.get("current_index", 0))

            while start_index < len(self.datasets):
                remaining_active = self.datasets[start_index:]
                name, cfg, split, field, _weight, _gated, kind = remaining_active[0]
                label = dataset_label(name, cfg)
                self.save_progress(phase=f"stream:{label}")

                ds_state = self.progress["datasets"].setdefault(
                    label,
                    {"done": False, "examples_seen": 0, "tokens": 0, "target": None, "restarts": 0},
                )

                if ds_state.get("done"):
                    log_event("dataset_skip_done", dataset=label)
                    start_index += 1
                    self.progress["current_index"] = start_index
                    self.save_progress()
                    continue

                target = ds_state.get("target")
                if target is None:
                    target = self.dataset_target(remaining_budget, remaining_active)
                    ds_state["target"] = int(target)
                    self.save_progress()

                if target <= 0:
                    ds_state["done"] = True
                    start_index += 1
                    self.progress["current_index"] = start_index
                    self.save_progress()
                    continue

                log_event("dataset_begin", dataset=label, target_tokens=int(target), kind=kind,
                          resume_examples=int(ds_state.get("examples_seen", 0)),
                          resume_tokens=int(ds_state.get("tokens", 0)))

                got = int(ds_state.get("tokens", 0))
                try:
                    for examples_seen, ex in self.iter_stream_restartable(name, cfg, split, label, ds_state):
                        if got >= target:
                            break
                        got += self.tokenize_to_buffer(text_from_example(ex, field, kind))
                        ds_state["examples_seen"] = int(examples_seen)
                        ds_state["tokens"] = int(got)
                        if len(self.write_buf) >= FLUSH_EVERY:
                            self.flush_buf(tmp_f)
                            log_event("progress", dataset=label, tokens=got, target=int(target),
                                      total_written=int(self.progress["total_written"]),
                                      examples=int(examples_seen))

                    self.flush_buf(tmp_f)
                    ds_state["tokens"] = int(got)
                    ds_state["done"] = True
                    # Free the (now useless) resume checkpoint.
                    ds_state.pop("hf_state_b64", None)
                    ds_state.pop("hf_state_base", None)
                    remaining_budget = max(0, remaining_budget - got)
                    log_event("dataset_done", dataset=label, tokens=got, remaining_budget=remaining_budget)
                    start_index += 1
                    self.progress["current_index"] = start_index
                    self.save_progress()
                except Exception as exc:
                    self.flush_buf(tmp_f)
                    total = int(self.progress.get("total_written", 0))
                    # A dataset that can't be streamed is only FATAL if we don't yet
                    # have enough data. If MIN_TOKENS is already met, skip it and
                    # finalize with what we have — one optional/gated source must not
                    # throw away a multi-hour, otherwise-complete tokenization.
                    if total >= MIN_TOKENS:
                        log_event("dataset_skipped", dataset=label, error=str(exc),
                                  total_written=total, min_tokens=MIN_TOKENS,
                                  note="source failed but MIN_TOKENS already met — skipping, not failing")
                        ds_state["done"] = True
                        ds_state["skipped"] = True
                        ds_state.pop("hf_state_b64", None)
                        ds_state.pop("hf_state_base", None)
                        start_index += 1
                        self.progress["current_index"] = start_index
                        self.save_progress(last_event=f"skipped:{label}")
                        continue
                    self.save_progress(last_event=f"dataset_failed:{label}")
                    raise

    # ---- custom chat -----------------------------------------------------
    def prepare_custom_chat(self):
        if self.progress.get("custom_done"):
            log_event("custom_skip_done")
            return
        if not self.custom_files:
            self.progress["custom_done"] = True
            self.save_progress()
            return

        self.save_progress(phase="custom")
        with open(self.TMP_PATH, "ab") as tmp_f:
            for fp in self.custom_files:
                label = f"custom:{Path(fp).name}"
                state = self.progress["datasets"].setdefault(
                    label, {"done": False, "tokens": 0, "examples_seen": 0}
                )
                if state.get("done"):
                    continue

                got = int(state.get("tokens", 0))
                seen = int(state.get("examples_seen", 0))
                global_seen = 0
                log_event("custom_begin", file=label, epochs=CUSTOM_CHAT_EPOCHS)

                for _epoch in range(CUSTOM_CHAT_EPOCHS):
                    for obj in read_jsonl(fp):
                        global_seen += 1
                        if global_seen <= seen:
                            continue
                        turns = normalize_turns(obj.get("messages", []))
                        roles = {r for r, _ in turns}
                        if "user" not in roles or "assistant" not in roles:
                            continue
                        got += self.tokenize_to_buffer(format_chat(turns))
                        state["tokens"] = int(got)
                        state["examples_seen"] = int(global_seen)
                        if len(self.write_buf) >= FLUSH_EVERY:
                            self.flush_buf(tmp_f)

                self.flush_buf(tmp_f)
                state["done"] = True
                self.save_progress()
                log_event("custom_done_file", file=label, tokens=got)

        self.progress["custom_done"] = True
        self.save_progress()

    # ---- finalize --------------------------------------------------------
    def copy_range(self, src_path, dst_path, start_tok, count_tok):
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
        log_event("wrote_split", path=str(dst_path), tokens=int(count_tok))

    def split_train_val(self):
        self.save_progress(phase="splitting")
        total_collected = int(self.progress.get("total_written", 0))
        per_dataset = {k: int(v.get("tokens", 0)) for k, v in self.progress.get("datasets", {}).items()}
        log_event("collection_summary", total=total_collected, target=TOTAL_TOKENS, per_dataset=per_dataset)

        if total_collected < MIN_TOKENS:
            self.write_failed(
                f"only {total_collected:,} tokens collected; MIN_TOKENS={MIN_TOKENS:,}. "
                f"Temp file + progress kept so the next run resumes.",
                fatal=False,
            )
            log_event("fatal", reason="below MIN_TOKENS", collected=total_collected, min_tokens=MIN_TOKENS)
            sys.exit(3)

        val_size = max(SEQ_LEN * 100, total_collected // 50)
        val_size = min(val_size, total_collected - SEQ_LEN)
        train_size = total_collected - val_size
        log_event("splitting", val_tokens=val_size, train_tokens=train_size)

        self.copy_range(self.TMP_PATH, self.VAL_PATH, 0, val_size)
        self.copy_range(self.TMP_PATH, self.TRAIN_PATH, val_size, train_size)

        try:
            self.TMP_PATH.unlink()
        except FileNotFoundError:
            pass

        self.progress["finalized"] = True
        self.save_progress(phase="done", last_event="finalized")
        try:
            self.FAILED_PATH.unlink()
        except FileNotFoundError:
            pass
        log_event("done", train=str(self.TRAIN_PATH), val=str(self.VAL_PATH), total_tokens=total_collected)

    # ---- failure reporting ----------------------------------------------
    def write_failed(self, reason, fatal=True):
        rec = {
            "result": "FAILED" if fatal else "INCOMPLETE",
            "reason": reason,
            "profile": PROFILE,
            "total_written": int(self.progress.get("total_written", 0)),
            "target": TOTAL_TOKENS,
            "min_tokens": MIN_TOKENS,
            "phase": self.progress.get("phase"),
            "current_index": self.progress.get("current_index"),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            with open(self.FAILED_PATH, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, sort_keys=True)
        except Exception:
            pass

    # ---- entrypoint ------------------------------------------------------
    def run(self):
        if (
            self.TRAIN_PATH.exists()
            and self.VAL_PATH.exists()
            and self.TRAIN_PATH.stat().st_size > 0
            and self.VAL_PATH.stat().st_size > 0
        ):
            log_event("already_done", train=str(self.TRAIN_PATH), val=str(self.VAL_PATH))
            return

        # Clear any stale failure marker from a previous attempt up front, so it
        # exists ONLY if THIS run fails. (Otherwise a resumed run that succeeds can
        # look failed when the orchestration re-syncs a leftover marker before the
        # finalize step removes it.)
        try:
            self.FAILED_PATH.unlink()
        except FileNotFoundError:
            pass

        self.save_progress(phase="starting", last_event="setup")
        self._setup()
        log_event("start", profile=PROFILE, total_tokens=TOTAL_TOKENS, min_tokens=MIN_TOKENS,
                  hf_token=bool(HF_TOKEN), stream_timeout=STREAM_TIMEOUT, max_stalls=MAX_STALLS,
                  max_restarts_no_progress=MAX_RESTARTS_NO_PROGRESS, checkpoint_every=CHECKPOINT_EVERY,
                  flush_every=FLUSH_EVERY, resume=RESUME_DATA_PREP)
        try:
            self.prepare_weighted_mix()
            self.prepare_custom_chat()
            self.split_train_val()
        except Exception as exc:  # noqa: BLE001
            self.write_failed(f"{type(exc).__name__}: {exc}")
            log_event("failed", error=repr(exc))
            raise


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    if PROFILE not in ("general", "coding"):
        log_event("fatal", reason=f"DATA_PROFILE must be general|coding (got {PROFILE!r})")
        sys.exit(1)
    DataPrep(output_dir).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_event("interrupted", note="progress kept; rerun to resume")
        sys.exit(130)
