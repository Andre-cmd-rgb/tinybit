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

## Current files

- `identity-tools-01.jsonl` — 442 identity + tool-use examples (AI-generated, validated).
- `identity-tools-02.jsonl` — 663 identity + tool-use examples (AI-generated, validated).
- `identity-tools-03.jsonl` — 240 identity + tool-use examples (AI-generated, validated).

Total: 1345 validated entries. The definitive micro run mixes these at `CUSTOM_CHAT_EPOCHS=50`.
