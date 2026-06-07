# Custom training datasets (drop folder)

Put your own curated chat data here as **JSONL** (one JSON object per line):

```json
{"messages":[{"role":"user","content":"who are you?"},{"role":"assistant","content":"I'm tinybit…"}]}
```

`scripts/prepare_data.sh` automatically tokenizes every `*.jsonl`/`*.txt` file in
this folder into the training mix:

- They are tokenized **last**, so the validation split (the head of the stream)
  stays the clean FineWeb-Edu distribution.
- Each file is repeated `CUSTOM_CHAT_EPOCHS` times (default **10**) and the tokens
  are added **on top of** `TOTAL_TOKENS`, so a small curated set is reliably seen
  without being drowned out by the 1.5B-token base mix.
- Generate more with the prompt in `prompts/tinybit-identity-tools-dataset.md`.

## Validate before training

```bash
python scripts/validate_chat_jsonl.py datasets/identity-tools-01.jsonl
```

It checks JSON validity, role/content structure, and that every `<|tool_call|>`
uses a real built-in tool with balanced markers. Fix any **hard errors** before
training (warnings are fine).

## Knobs

| env var | default | meaning |
|---|---|---|
| `CUSTOM_CHAT_DIR` | `datasets` | folder scanned for custom files |
| `CUSTOM_CHAT_EPOCHS` | `10` | how many times each file is repeated (`0` disables) |

## Note on the GCP pipeline

Files here are uploaded with the repo and tokenized on the VM. Because tokenized
data is **cached per-run** in the bucket, adding a new file to an *existing* run
needs a re-tokenize: relaunch with `RESET_RUN=1` (or `FORCE_DATA=1`). A fresh
`RUN_ID` always re-tokenizes.

## Tool / no-tool balance (IMPORTANT)

tinybit is tiny (~50M). If a large fraction of curated examples call a tool, the
model learns that calling a tool is the *default* reply and fires them on almost
anything (greetings, general questions) with garbage arguments — exactly what the
first definitive run did. Two rules:

1. **Keep tool examples a MINORITY** of the custom set — aim for ≤ ~20% with a
   `<|tool_call|>`. Most lines should be answered directly in words, including
   "tool-shaped" prompts (numbers, lists, dates) that don't actually need a tool.
   `validate_chat_jsonl.py` prints the tool-call % — watch it.
2. **Keep `CUSTOM_CHAT_EPOCHS` LOW (≈8–10), not 50.** Repeating the narrow custom
   set 50× memorizes the tool strings into an attractor. The base mix already has
   ~22% diverse, non-repeated no-tool instruction data (OpenHermes/dolphin).

## Current files

- `identity-tools-01.jsonl` — 442 identity + tool-use examples (AI-generated, validated).
- `identity-tools-02.jsonl` — 663 identity + tool-use examples (AI-generated, validated).
- `identity-tools-03.jsonl` — 240 identity + tool-use examples (AI-generated, validated).
- `chat-notools-04.jsonl` — 90 no-tool examples (identity, general help, and
  "tool-shaped but answered directly") added 2026-06-07 to dilute the tool ratio.
- `chat-lookup-05.jsonl` — 58 examples teaching the `lookup` tool (facts →
  call lookup → answer from the result; not-found → admit it, don't bluff) plus
  correct usage of the other tools. The `<|tool_result|>` text mirrors what the
  real tools return, so training and inference agree. Added 2026-06-07.
- `chat-summary-06.jsonl` — 41 no-tool examples for the headline skills:
  summarising, explaining/ELI5, paraphrasing, and reading comprehension. Added
  2026-06-07 alongside the language-first base-mix retune (no code).

⚠️ The `identity-tools-0{1,2,3}` files are **~54% tool calls** — too tool-heavy on
their own (see balance rules above). For the next run, prefer regenerating them
with the updated `prompts/tinybit-identity-tools-dataset.md` (now ~15% tools) so
the whole set lands near a ~15–20% tool ratio, and train with
`CUSTOM_CHAT_EPOCHS≈8` (NOT 50, which caused the over-firing).
