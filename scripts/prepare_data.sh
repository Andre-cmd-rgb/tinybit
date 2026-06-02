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
#   CUSTOM_CHAT_DIR    dir of your own {"messages":[...]} JSONL/TXT files to mix in
#                      (default: datasets/). Validate them with
#                      scripts/validate_chat_jsonl.py first.
#   CUSTOM_CHAT_EPOCHS times to repeat each custom file (default 10; 0 disables).
#                      Custom tokens are tokenized LAST and added on top of
#                      TOTAL_TOKENS, so the val split stays the FineWeb-Edu head.
#
# Profiles (the ONLY difference between the general and coding model families):
#   general — natural-language assistant mix tuned for a SMALL model: FineWeb-Edu
#             (educational web) + Cosmopedia v2 (synthetic textbooks) + TinyStories
#             (coherence) + OpenHermes/dolphin chat + a little code. There is NO
#             raw Wikipedia: a ~50M model cannot store encyclopedic facts, so it
#             only mimics Wikipedia's proper-noun register and hallucinates
#             band/album/biography trivia. Clean pedagogical/narrative prose
#             teaches it to explain and stay coherent instead (cf. SmolLM,
#             TinyStories, Phi "textbooks are all you need").
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/prepare_data.py" "$OUTPUT_DIR"
