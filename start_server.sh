#!/usr/bin/env bash
# Launch the TRIBE v2 WebSocket server in a detached tmux session on the pod.
# Idempotent — re-running attaches to the existing session if one is up.
#
# Usage on the pod (after setup_tribev2.sh has finished):
#     bash /workspace/tribev2/start_server.sh
#
# Tail logs:    tail -f /workspace/server.log
# Attach tmux:  tmux attach -t tribe
# Stop server:  tmux kill-session -t tribe

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
SERVER="$WORKDIR/tribev2/server.py"
LOG="$WORKDIR/server.log"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

if [ ! -f "$SERVER" ]; then
  echo "server.py not found at $SERVER — run setup_tribev2.sh first" >&2
  exit 1
fi

apt-get install -y tmux >/dev/null 2>&1 || true

if tmux has-session -t tribe 2>/dev/null; then
  log "tmux session 'tribe' already exists. Attach with: tmux attach -t tribe"
  exit 0
fi

# Persist the env across tmux session
ENV_LINE='export HF_HOME=/workspace/.cache/huggingface TRIBE_CACHE=/workspace/.cache/tribev2'
log "Starting server in detached tmux session 'tribe'"
tmux new -d -s tribe "$ENV_LINE && python $SERVER 2>&1 | tee $LOG"

log "Server starting. First boot does:"
log "  1) load TRIBE checkpoint        (~15s)"
log "  2) fetch HCP-MMP parcellation   (~1-2 min, only on a fresh pod)"
log "  3) bake AO + warmup encoders    (~60s)"
log "  4) ready on 0.0.0.0:8000"
log ""
log "Watch progress:"
log "  tail -f $LOG"
log "or attach the tmux session:"
log "  tmux attach -t tribe   (detach with Ctrl-b d)"
