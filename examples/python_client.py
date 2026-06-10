#!/usr/bin/env python3
"""Example tinybit data-API client (Python, stdlib only).

The tinybit data API is a FILE contract (see INTEGRATIONS.md): any app that
can spawn a process or append a line to a file can push user data (smartwatch
metrics, health, app counters) into tinybit's view. This example shows both
ways and a tiny reusable helper you can copy into your own app.

Usage:
    python examples/python_client.py            # push a fake smartwatch sample
    python examples/python_client.py --direct   # same, via direct file append
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")  # the same --data-dir your `tinybit chat` uses


def push_events(source: str, events, data_dir: Path = DATA_DIR) -> str:
    """Recommended: push events through `tinybit ingest`.

    The CLI validates names/timestamps, appends atomically, and updates the
    latest.json snapshot. `events` is a list of dicts:
        {"metric": "heart_rate", "value": 61, "unit": "bpm", "ts": "...Z"}
    ("ts" defaults to now; "unit"/"tags" are optional.)
    """
    result = subprocess.run(
        ["tinybit", "ingest", "--source", source, "--data-dir", str(data_dir)],
        input=json.dumps(events),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tinybit ingest failed: {result.stderr.strip()}")
    return result.stdout.strip()


def push_events_direct(source: str, events, data_dir: Path = DATA_DIR) -> None:
    """Fallback: the file contract IS the API — append complete JSON lines.

    Use this when shelling out isn't practical (e.g. a daemon). Rules
    (INTEGRATIONS.md): one complete JSON object per line, RFC3339 "ts",
    lowercase metric names. tinybit's readers skip malformed lines, but
    `latest.json` is only refreshed by `tinybit ingest`, so prefer that path
    when you can.
    """
    src_dir = data_dir / "integrations" / source
    src_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(src_dir / "events.jsonl", "a", encoding="utf-8") as f:
        for e in events:
            e.setdefault("ts", now)
            f.write(json.dumps(e) + "\n")


def main() -> None:
    # A fake smartwatch sample — in a real plug-in, read your device's SDK/
    # export instead.
    sample = [
        {"metric": "heart_rate", "value": 61, "unit": "bpm"},
        {"metric": "steps", "value": 8204},
        {"metric": "sleep_hours", "value": 7.5, "unit": "h",
         "ts": datetime.now(timezone.utc).isoformat()},
    ]
    if "--direct" in sys.argv:
        push_events_direct("watch", sample)
        print(f"appended {len(sample)} event(s) directly to "
              f"{DATA_DIR / 'integrations' / 'watch' / 'events.jsonl'}")
    else:
        print(push_events("watch", sample))
    print("Ask tinybit: `tinybit chat` → \"what's my heart rate?\" "
          "(the user_data tool reads this store)")


if __name__ == "__main__":
    main()
