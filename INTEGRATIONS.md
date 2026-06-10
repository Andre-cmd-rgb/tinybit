# The tinybit data API (integrations)

tinybit's answer to "give the assistant a full view of the user" is **not** to
train facts into the model — it's to let apps **plug data in locally** and let
the model **fetch and reason over it** with the `user_data` tool.

The API is deliberately a **file contract plus a CLI**, not a server (tinybit
is local-first; see design decision 19/25 in CLAUDE.md). Any language that can
spawn a process or append a line to a file can integrate: a smartwatch
exporter, a scale, a script that counts your git commits — anything.

```
your app ──(JSON)──▶ tinybit ingest ──▶ data/integrations/<source>/   ◀── user_data tool ◀── the model
```

---

## Quick start

```bash
# Push a reading (any app, any language — this is the whole API):
echo '{"metric":"heart_rate","value":61,"unit":"bpm"}' | tinybit ingest --source watch

# Batch (JSON array or JSONL also accepted):
echo '[{"metric":"steps","value":8204},{"metric":"sleep_hours","value":7.5,"unit":"h"}]' \
  | tinybit ingest --source watch

# Then ask in chat:
tinybit chat ...
> what's my heart rate?
```

Working client examples: [`examples/python_client.py`](examples/python_client.py)
(stdlib only) and [`examples/rust_client/`](examples/rust_client/) (standalone
crate — copy it as your starting point).

> The model only *calls* `user_data` reliably after a retrain that includes
> `datasets/chat-userdata-08.jsonl` (see TRAINING.md's next-run checklist).
> The tool itself works immediately — try it raw with `--tools always`.

---

## Event schema

One event = one JSON object:

```json
{"ts": "2026-06-10T08:31:00Z", "metric": "heart_rate", "value": 61, "unit": "bpm", "tags": {"activity": "rest"}}
```

| field | required | rules |
|---|---|---|
| `metric` | yes | `[a-z0-9_]{1,64}` (uppercase is normalized down) |
| `value` | yes | number, bool, or string ≤ 256 chars (metrics, not documents) |
| `ts` | no | RFC3339, `YYYY-MM-DD`, or unix seconds; **defaults to now** |
| `unit` | no | short string shown back to the user (`bpm`, `kg`, `h`) |
| `tags` | no | flat JSON object, free-form |

`tinybit ingest` accepts a single object, a JSON array, or JSONL (one object
per line) on stdin or via `--file`. Invalid events are skipped and counted;
the command fails only if **no** event was valid.

## Store layout

```
data/integrations/
  <source>/                 # [a-z0-9_-]{1,32}, e.g. "watch", "scale"
    events.jsonl            # append-only log (auto-rotated past ~10 MB)
    latest.json             # {"<metric>": {"ts", "value", "unit"}} snapshot
    meta.json               # {"source", "schema_version": 1, "created"}
```

`--data-dir` (default `data/`) must match the one `tinybit chat` runs with.

### The write contract (for direct writers)

`tinybit ingest` is the recommended front door — it validates, appends, and
rewrites `latest.json` via temp-file + atomic rename. But **direct appends are
explicitly allowed**: that's what makes the API language-agnostic.
If you write `events.jsonl` yourself:

1. append **complete lines** — one valid JSON event per line, `\n`-terminated;
2. use append mode (`O_APPEND`), so concurrent writers interleave whole lines;
3. don't edit or reorder existing lines.

Readers are tolerant by design: malformed lines are **skipped and counted,
never fatal**. Note `latest.json` is only refreshed by `tinybit ingest` /
the `IntegrationsStore` API — direct writers still get range queries
(`user_data` `range` reads the logs), just a possibly stale `latest` until
the next ingest.

Rust apps can also link `tinybit-tools` and use
`tinybit_tools::integrations::IntegrationsStore` directly (same code path as
the CLI).

---

## How the model sees it: the `user_data` tool

Registered alongside the other built-ins; schema:

```json
{"action": "latest|range|sources", "source": "string?", "metric": "string?", "since": "string?", "until": "string?"}
```

| action | result (one compact line per item) |
|---|---|
| `latest` | `heart_rate: 61 bpm at 2026-06-10T08:31Z (watch)` |
| `range` | `steps 2026-06-03→2026-06-10: count=14 min=4102 max=11873 mean=8204 last=9120` |
| `sources` | `Sources: watch (heart_rate, steps), scale (weight)` |
| (no data) | `No data for "blood_pressure".` — an honest miss, never invented numbers |

`since`/`until` accept RFC3339, `YYYY-MM-DD`, unix seconds, and the relative
forms a small model can reliably emit: `today`, `yesterday`, `now`, `7d`,
`12h`. `range` defaults to `since=7d`, `until=now`.

In `chat`, the tool gate arms `user_data` for questions that pair a metric
word (heart rate, steps, sleep, …) with an inquiry/possessive cue ("what's
**my** heart rate", "how many steps **today**") — and stays quiet for
incidental mentions ("weight loss is hard").

## Design notes

- **Files, not sockets.** Greppable, debuggable, no daemon, no port, works
  over syncthing/git/rsync. A server would violate tinybit's local-first
  contract.
- **JSONL, not SQLite.** Third-party writers in any language can append a
  line; nobody needs a driver. The volumes here (personal metrics) never need
  an index.
- **The model interprets, the store remembers.** Training data teaches the
  model to restate and reason over fetched values ("your resting HR averaged
  61 this week") and to admit when there's no data — never to memorize or
  invent the numbers.
