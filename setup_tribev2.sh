#!/usr/bin/env bash
# Set up and smoke-test facebook/tribev2 on a RunPod box (or any CUDA Linux host).
#
# Prereqs (do these on the HF website BEFORE running):
#   1) Request access to https://huggingface.co/meta-llama/Llama-3.2-3B
#   2) Create a READ token at https://huggingface.co/settings/tokens
#
# Usage on the pod:
#   scp this file to the pod, then:
#     apt-get install -y tmux && tmux new -s tribe   # SSH on RunPod web idle-times at ~5 min
#     export HF_TOKEN=hf_xxx                          # optional; otherwise script prompts
#     bash setup_tribev2.sh
#   Detach with Ctrl-b d, reattach with `tmux attach -t tribe`.
#
# Re-running is safe: each step is skipped if already done.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/tribev2"
CACHE_DIR="$WORKDIR/.cache/huggingface"
TRIBE_CACHE="$WORKDIR/.cache/tribev2"
EXTRAS="${EXTRAS:-plotting}"   # set EXTRAS=training for the full dev install, or empty for inference-only

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. system deps
# ---------------------------------------------------------------------------
log "Installing system deps (ffmpeg, git, git-lfs)"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends ffmpeg git git-lfs ca-certificates >/dev/null
  git lfs install --skip-repo
else
  echo "apt-get not found; install ffmpeg, git, git-lfs manually." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. python sanity check (TRIBE requires 3.11+)
# ---------------------------------------------------------------------------
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "Python: $PY_VERSION"
python3 - <<'PY'
import sys
assert sys.version_info >= (3, 11), f"TRIBE v2 requires Python >= 3.11, got {sys.version.split()[0]}"
PY

# ---------------------------------------------------------------------------
# 3. clone repo
# ---------------------------------------------------------------------------
mkdir -p "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  log "Cloning facebookresearch/tribev2 -> $REPO_DIR"
  git clone https://github.com/facebookresearch/tribev2.git "$REPO_DIR"
else
  log "Repo already at $REPO_DIR (skipping clone). Pulling latest."
  git -C "$REPO_DIR" pull --ff-only || true
fi

# ---------------------------------------------------------------------------
# 4. pip install
# ---------------------------------------------------------------------------
log "pip install -e .[${EXTRAS}]"
cd "$REPO_DIR"
if [ -n "$EXTRAS" ]; then
  pip install --upgrade pip
  pip install -e ".[${EXTRAS}]"
else
  pip install --upgrade pip
  pip install -e .
fi

# uv / uvx — TRIBE shells out to `uvx whisperx` for text-input transcription.
log "Installing uv (provides uvx)"
pip install --upgrade uv
command -v uvx >/dev/null || { echo "uvx still missing after 'pip install uv'" >&2; exit 1; }

# WebSocket server deps (server.py)
log "Installing FastAPI / uvicorn / websockets for the WS server"
pip install --upgrade fastapi 'uvicorn[standard]' websockets

# ---------------------------------------------------------------------------
# 5. caches on the persistent volume
# ---------------------------------------------------------------------------
mkdir -p "$CACHE_DIR" "$TRIBE_CACHE"
export HF_HOME="$CACHE_DIR"
if ! grep -q "HF_HOME=$CACHE_DIR" "$HOME/.bashrc" 2>/dev/null; then
  printf '\nexport HF_HOME=%s\n' "$CACHE_DIR" >> "$HOME/.bashrc"
fi
log "HF_HOME=$HF_HOME"

# ---------------------------------------------------------------------------
# 6. HF auth (needs gated LLaMA-3.2-3B access)
# ---------------------------------------------------------------------------
# huggingface_hub >=1.0 renamed the CLI to `hf`; old `huggingface-cli` is a no-op.
HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  # If $HF_TOKEN is set, ALWAYS use it — overwrite any stale cached token
  # from a previous run (e.g. a token that lacked gated LLaMA access).
  rm -f "$CACHE_DIR/token" "$HOME/.cache/huggingface/token"
  log "Logging into Hugging Face via \$HF_TOKEN ($HF_CLI)"
  if [ "$HF_CLI" = "hf" ]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
  else
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  fi
elif [ -f "$CACHE_DIR/token" ] || [ -f "$HOME/.cache/huggingface/token" ]; then
  log "HF token already cached and no \$HF_TOKEN override, reusing it."
else
  log "No HF_TOKEN env var; running interactive login. Paste your read token."
  if [ "$HF_CLI" = "hf" ]; then
    hf auth login
  else
    huggingface-cli login
  fi
fi

# Sanity-check the token actually has gated LLaMA-3.2-3B access before
# burning ~15 GB of weight downloads on a doomed run.
log "Verifying gated access to meta-llama/Llama-3.2-3B"
if ! hf download meta-llama/Llama-3.2-3B config.json --revision main >/dev/null 2>&1; then
  echo "ERROR: this HF token cannot access meta-llama/Llama-3.2-3B." >&2
  echo "  1) Visit https://huggingface.co/meta-llama/Llama-3.2-3B and request access." >&2
  echo "  2) Once approved, create a fresh READ token at https://huggingface.co/settings/tokens" >&2
  echo "  3) Re-run with: export HF_TOKEN=hf_xxx && bash setup_tribev2.sh" >&2
  exit 1
fi
log "Gated access OK."

# ---------------------------------------------------------------------------
# 7. CUDA visibility
# ---------------------------------------------------------------------------
log "CUDA / GPU check"
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
else:
    print("WARNING: no CUDA. TRIBE will be unusably slow on CPU.")
PY

# ---------------------------------------------------------------------------
# 8. smoke test: text-only inference (downloads ~15-25 GB on first run)
# ---------------------------------------------------------------------------
log "Smoke test: text-only prediction (first run downloads weights)"
mkdir -p "$WORKDIR/io"
SAMPLE="$WORKDIR/io/sample.txt"
[ -f "$SAMPLE" ] || printf 'The quick brown fox jumps over the lazy dog.\n' > "$SAMPLE"

TRIBE_CACHE="$TRIBE_CACHE" SAMPLE="$SAMPLE" python3 - <<'PY'
import os
from tribev2 import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"]
)
df = model.get_events_dataframe(text_path=os.environ["SAMPLE"])
preds, segments = model.predict(events=df)
print("preds shape:", preds.shape, "segments:", len(segments))
PY

log "Done. Repo: $REPO_DIR  Caches: $CACHE_DIR, $TRIBE_CACHE"
log "Next: see run_image_text.py for the image+text inference recipe."
