#!/usr/bin/env bash
# Move everything that should survive a RunPod Stop onto /workspace, and
# write /workspace/on_resume.sh so you have a one-shot resume command.
# Idempotent — safe to re-run.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"

log() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }

# -- 1. mne data (HCP-MMP parcellation, fsaverage subject) --------------
if [ -d /root/mne_data ] && [ ! -L /root/mne_data ]; then
  log "Moving /root/mne_data -> $WORKDIR/mne_data (using rsync; ignores ownership-preserve errors on volume disk)"
  mkdir -p "$WORKDIR/mne_data"
  rsync -a --no-owner --no-group --no-perms --remove-source-files \
    /root/mne_data/ "$WORKDIR/mne_data/"
  find /root/mne_data -depth -type d -empty -delete 2>/dev/null || true
  rm -rf /root/mne_data
fi
ln -sfn "$WORKDIR/mne_data" /root/mne_data

# -- 2. HF token on the volume ------------------------------------------
mkdir -p "$WORKDIR/.cache/huggingface"
if [ ! -s "$WORKDIR/.cache/huggingface/token" ] && [ -s /root/.cache/huggingface/token ]; then
  log "Copying HF token to $WORKDIR/.cache/huggingface/"
  cp /root/.cache/huggingface/token "$WORKDIR/.cache/huggingface/token"
fi
# Also copy the new "stored_tokens" file used by huggingface_hub >=1.0
if [ ! -s "$WORKDIR/.cache/huggingface/stored_tokens" ] && [ -s /root/.cache/huggingface/stored_tokens ]; then
  cp /root/.cache/huggingface/stored_tokens "$WORKDIR/.cache/huggingface/stored_tokens"
fi

# -- 3. environment file sourced on resume -------------------------------
log "Writing $WORKDIR/env.sh"
cat > "$WORKDIR/env.sh" <<'SH'
export HF_HOME=/workspace/.cache/huggingface
export TRIBE_CACHE=/workspace/.cache/tribev2
export MNE_DATA=/workspace/mne_data
SH

# -- 4. pip freeze for fast reinstall on resume --------------------------
log "Writing $WORKDIR/pip-freeze.txt"
pip freeze > "$WORKDIR/pip-freeze.txt" || true

# -- 5. on_resume.sh: re-install packages and start server in tmux -------
log "Writing $WORKDIR/on_resume.sh"
cat > "$WORKDIR/on_resume.sh" <<'SH'
#!/usr/bin/env bash
# Run on the pod after a Stop -> Resume to bring the WebSocket server back.
# Idempotent.
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace}"
source "$WORKDIR/env.sh"
ln -sfn "$WORKDIR/mne_data" /root/mne_data 2>/dev/null || true

# Reinstall pip deps (model weights / mne data already on the volume)
apt-get install -y tmux ffmpeg git git-lfs >/dev/null 2>&1 || true
pip install --upgrade pip >/dev/null
pip install -e "$WORKDIR/tribev2[plotting]"
pip install uv fastapi 'uvicorn[standard]' websockets

# Drop into a tmux session running the server
if tmux has-session -t tribe 2>/dev/null; then
  echo "tmux session 'tribe' already exists — attach with: tmux attach -t tribe"
  exit 0
fi
tmux new -d -s tribe "source $WORKDIR/env.sh && python $WORKDIR/tribev2/server.py 2>&1 | tee $WORKDIR/server.log"
echo "Started server in tmux session 'tribe'."
echo "  Tail logs:  tail -f $WORKDIR/server.log"
echo "  Attach:     tmux attach -t tribe"
SH
chmod +x "$WORKDIR/on_resume.sh"

# -- 6. summary ----------------------------------------------------------
log "Done. Volume size on disk:"
du -sh "$WORKDIR" 2>/dev/null || true
log "Now safe to Stop the pod from the RunPod UI."
log "After Resume, run:   bash /workspace/on_resume.sh"
