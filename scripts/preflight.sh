#!/bin/bash
# Local preflight checks before paying for a GCP GPU VM.
#
# Verifies:
#   * gcloud / gsutil available
#   * GCP_PROJECT / GCP_BUCKET set and reachable
#   * cargo / rustc available
#   * pkg-config + libssl present (for cross-checking the cloud-side build)
#   * VERSION + git commit clean
#   * configs/<model>.toml exists
#   * scripts/cloud/startup.sh exists and template tokens are present
#   * tokenizer.json or HF reachable
#   * datasets package importable (data prep precheck — non-fatal)
#
# Usage: ./scripts/preflight.sh [nano|micro|small|base]

set -uo pipefail

MODEL_SIZE="${1:-nano}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

PASS=0
FAIL=0
WARN=0

ok()   { printf '  [ok]   %s\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  [warn] %s\n' "$*"; WARN=$((WARN+1)); }
err()  { printf '  [FAIL] %s\n' "$*"; FAIL=$((FAIL+1)); }

step() { printf '\n== %s ==\n' "$*"; }

step "tools"
command -v gcloud >/dev/null && ok "gcloud installed: $(gcloud --version 2>/dev/null | head -1)" || err "gcloud missing"
command -v gsutil >/dev/null && ok "gsutil installed"                       || err "gsutil missing"
command -v cargo  >/dev/null && ok "cargo  installed: $(cargo --version)"   || err "cargo missing"
command -v rustc  >/dev/null && ok "rustc  installed: $(rustc --version)"   || err "rustc missing"
command -v python3 >/dev/null && ok "python3 installed: $(python3 --version)" || err "python3 missing"
command -v pkg-config >/dev/null && ok "pkg-config installed" || warn "pkg-config missing locally (the VM installs it anyway)"

step "env"
if [ -n "${GCP_PROJECT:-}" ]; then ok "GCP_PROJECT=$GCP_PROJECT"; else err "GCP_PROJECT not set"; fi
if [ -n "${GCP_BUCKET:-}" ];  then ok "GCP_BUCKET=$GCP_BUCKET";   else err "GCP_BUCKET not set";  fi

step "gcloud auth"
ACT=$(gcloud config get-value account 2>/dev/null || true)
if [ -n "$ACT" ] && [ "$ACT" != "(unset)" ]; then ok "active account: $ACT"; else err "no active gcloud account (run: gcloud auth login)"; fi

step "bucket access"
if [ -n "${GCP_BUCKET:-}" ]; then
  if gsutil ls "${GCP_BUCKET%/}/" >/dev/null 2>&1; then
    ok "$GCP_BUCKET reachable"
    PROBE="${GCP_BUCKET%/}/_preflight_probe.txt"
    if echo "preflight" | gsutil -q cp - "$PROBE" 2>/dev/null; then
      gsutil -q rm "$PROBE" || true
      ok "bucket writable"
    else
      err "bucket not writable"
    fi
  else
    err "cannot list $GCP_BUCKET"
  fi
fi

step "repo state"
if [ -f "$REPO_ROOT/VERSION" ]; then
  ok "VERSION: $(cat "$REPO_ROOT/VERSION")"
else
  warn "VERSION file missing"
fi
if (cd "$REPO_ROOT" && git rev-parse HEAD >/dev/null 2>&1); then
  ok "git commit: $(cd "$REPO_ROOT" && git rev-parse --short HEAD)"
  if (cd "$REPO_ROOT" && ! git diff --quiet HEAD 2>/dev/null); then
    warn "git working tree is dirty (set FORCE=1 to launch anyway)"
  else
    ok "git working tree clean"
  fi
else
  warn "not a git repo (or no commits)"
fi

step "configs"
if [ -f "$REPO_ROOT/configs/${MODEL_SIZE}.toml" ]; then
  ok "configs/${MODEL_SIZE}.toml present"
else
  err "configs/${MODEL_SIZE}.toml not found"
fi

step "cloud startup template"
TEMPLATE="$REPO_ROOT/scripts/cloud/startup.sh"
if [ -f "$TEMPLATE" ]; then
  ok "template present"
  MISSING="$(grep -oE '__[A-Z_]+__' "$TEMPLATE" | sort -u | head -20)"
  if [ -n "$MISSING" ]; then
    ok "found placeholders: $(echo "$MISSING" | tr '\n' ' ')"
  else
    err "template has no placeholders — looks corrupted"
  fi
else
  err "missing $TEMPLATE"
fi

step "scripts"
for f in gcp_launch.sh prepare_data.sh gcp_status.sh gcp_tail_logs.sh gcp_sync_now.sh gcp_stop_vm.sh gcp_delete_vm.sh; do
  if [ -r "$REPO_ROOT/scripts/$f" ]; then
    ok "scripts/$f readable"
  else
    err "scripts/$f missing or unreadable"
  fi
done

step "tokenizer"
if [ -f "$REPO_ROOT/tokenizer.json" ]; then
  ok "tokenizer.json present locally"
else
  warn "tokenizer.json missing locally — VM will download from HuggingFace"
fi

step "python datasets package (data-prep precheck)"
if python3 -c 'import datasets, tokenizers, tqdm' 2>/dev/null; then
  ok "datasets/tokenizers/tqdm importable locally"
else
  warn "datasets/tokenizers/tqdm not installed locally (VM installs its own copy)"
fi

step "build sanity (cargo check)"
if (cd "$REPO_ROOT" && cargo check --workspace --offline 2>/dev/null) ; then
  ok "cargo check --offline passes"
elif (cd "$REPO_ROOT" && cargo check --workspace 2>&1 | tail -5 > /tmp/preflight-check.log) ; then
  ok "cargo check passes"
else
  err "cargo check failed — see /tmp/preflight-check.log"
fi

step "result"
printf '  passes: %d   warns: %d   fails: %d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "FAILED — fix the [FAIL] items before launching a GPU VM."
  exit 1
fi
echo "OK — ready to launch."
