#!/usr/bin/env python3
"""Offline tests for prepare_data.py's stall/kill/restart/resume machinery.

Uses the TINYBIT_FAKE_STREAM seam (no network, no HF, no real tokenizer) to
prove, deterministically:

  1. a streaming "wedge" is detected, the child process is killed, the dataset
     is restarted, resumes via state_dict, and the run completes with correct
     train.bin / val.bin and NO failure marker;
  2. a *permanent* wedge is bounded — the run gives up after
     MAX_RESTARTS_NO_PROGRESS, writes prepare_FAILED.json, and exits non-zero.

Run:  python scripts/test_prepare_data.py
Exit: 0 = all passed, non-zero = a check failed.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARE = HERE / "prepare_data.py"


def run_case(env_extra, output_dir, timeout):
    env = dict(os.environ)
    env.update({
        "TINYBIT_FAKE_STREAM": "1",
        "PYTHONUNBUFFERED": "1",
        # Custom chat dir off so the test is purely the streaming machinery.
        "CUSTOM_CHAT_DIR": "/nonexistent-tinybit-test",
        "CUSTOM_CHAT_EPOCHS": "0",
        "RESUME_DATA_PREP": "0",
    })
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(PREPARE), str(output_dir)],
        env=env, capture_output=True, text=True, timeout=timeout,
    )
    return proc


def test_transient_wedge_recovers():
    print("\n=== TEST 1: transient wedge -> kill -> restart -> resume -> complete ===")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        flag = d / "hang.flag"
        proc = run_case({
            "TINYBIT_FAKE_N": "4000",
            "TINYBIT_FAKE_HANG_AT": "2000",       # wedge halfway, once
            "TINYBIT_FAKE_HANG_FLAG": str(flag),
            "TOTAL_TOKENS": "200000",             # < what 4000 fake examples yield
            "MIN_TOKENS": "1",
            "SEQ_LEN": "64",
            "STREAM_TIMEOUT": "3",
            "MAX_STALLS": "1",
            "FLUSH_EVERY": "1000",
            "CHECKPOINT_EVERY": "500",
        }, d, timeout=120)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-2000:])
            raise AssertionError(f"expected success, got rc={proc.returncode}")
        train, val = d / "train.bin", d / "val.bin"
        assert train.exists() and train.stat().st_size > 0, "train.bin missing/empty"
        assert val.exists() and val.stat().st_size > 0, "val.bin missing/empty"
        assert not (d / "prepare_FAILED.json").exists(), "unexpected failure marker"
        # Must have actually exercised the restart path.
        assert '"event": "restart"' in proc.stdout or '"event": "stall"' in proc.stdout, \
            "stall/restart path was not exercised"
        assert '"event": "done"' in proc.stdout, "no done event"
        # Exact-count check: each of the 4000 fake examples is 20 words ->
        # 21 fake-token ids + EOS = 22 tokens. A correct resume (no dup, no loss)
        # yields exactly 4000*22 = 88000 tokens across train+val.
        toks = (train.stat().st_size + val.stat().st_size) // 4
        expected = 4000 * 22
        assert toks == expected, f"token count {toks} != expected {expected} (dup or loss on resume!)"
        print(f"PASS: recovered from wedge with EXACT token count {toks:,} "
              f"(train={train.stat().st_size // 4:,}, val={val.stat().st_size // 4:,}) — no dup, no loss")


def test_permanent_wedge_fails_loudly():
    print("\n=== TEST 2: permanent wedge -> bounded restarts -> FAILED marker -> non-zero exit ===")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proc = run_case({
            "TINYBIT_FAKE_N": "4000",
            "TINYBIT_FAKE_HANG_AT": "0",          # wedge immediately, forever (no flag)
            "TOTAL_TOKENS": "200000",
            "MIN_TOKENS": "1",
            "SEQ_LEN": "64",
            "STREAM_TIMEOUT": "2",
            "MAX_STALLS": "1",
            "MAX_RESTARTS_NO_PROGRESS": "2",
        }, d, timeout=120)
        print(proc.stdout[-2000:])
        if proc.returncode == 0:
            raise AssertionError("expected non-zero exit on permanent wedge")
        failed = d / "prepare_FAILED.json"
        assert failed.exists(), "prepare_FAILED.json was not written"
        rec = json.loads(failed.read_text())
        assert rec["result"] == "FAILED", f"unexpected marker: {rec}"
        assert not (d / "train.bin").exists(), "train.bin should not exist on failure"
        print(f"PASS: failed loudly after bounded restarts; marker reason: {rec['reason'][:80]}...")


def test_failing_dataset_skipped_when_min_met():
    print("\n=== TEST 3: dataset fails AFTER MIN_TOKENS met -> skip + finalize (not fail) ===")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        proc = run_case({
            "TINYBIT_FAKE_N": "8000",
            "TINYBIT_FAKE_HANG_AT": "2000",   # produce 2000 examples, then wedge forever
            "TOTAL_TOKENS": "10000000",       # target far above what 2000 examples yield
            "MIN_TOKENS": "1",                # ...but MIN is already met after a few examples
            "SEQ_LEN": "64",
            "STREAM_TIMEOUT": "2",
            "MAX_STALLS": "1",
            "MAX_RESTARTS_NO_PROGRESS": "2",
            "CHECKPOINT_EVERY": "500",
        }, d, timeout=120)
        print(proc.stdout[-1500:])
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1500:])
            raise AssertionError(f"expected success (skip+finalize), got rc={proc.returncode}")
        assert '"event": "dataset_skipped"' in proc.stdout, "dataset was not skipped after MIN met"
        train, val = d / "train.bin", d / "val.bin"
        assert train.exists() and val.exists(), "train/val not produced after skip"
        assert not (d / "prepare_FAILED.json").exists(), "should not have failed"
        toks = (train.stat().st_size + val.stat().st_size) // 4
        # ~2000 examples * 22 tokens were collected before the permanent wedge.
        assert 2000 * 22 * 0.5 < toks <= 2000 * 22 + 22, f"unexpected token count {toks}"
        print(f"PASS: failing trailing dataset skipped, finalized with {toks:,} tokens")


if __name__ == "__main__":
    test_transient_wedge_recovers()
    test_permanent_wedge_fails_loudly()
    test_failing_dataset_skipped_when_min_met()
    print("\nALL TESTS PASSED")
