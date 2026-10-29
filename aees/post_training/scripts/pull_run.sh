#!/usr/bin/env bash
# On the laptop: copy one run's tracker inputs from the box, laid out like the box's $DATA, so the
# tracker also runs locally on them:  python3 monitor/server.py --run <run> --data <dest>
# Copies the Miles event log, every trial's result.json (training and eval), the checkpoint folder
# names (no weights), host samples, and a status.json snapshot. A few MB; transcripts only on request.
#
#   bash scripts/pull_run.sh ubuntu@<box> grpo_pilot1 [dest=runs]
#   WITH_TRAJECTORIES=1   also copy each trial's agent/trajectory.json (MBs per trial)
set -euo pipefail
HOST=${1:?usage: pull_run.sh user@box RUN [dest]}; RUN=${2:?usage: pull_run.sh user@box RUN [dest]}
DEST=${3:-runs}; DATA=${DATA:-/data}
TRIAL_FILES=(--include='*/' --include='result.json')
[ "${WITH_TRAJECTORIES:-0}" = 1 ] && TRIAL_FILES+=(--include='trajectory.json')
mkdir -p "$DEST/ckpt/$RUN" "$DEST/trials/$RUN" "$DEST/trials/eval/$RUN" "$DEST/monitor/$RUN"
rsync -az "$HOST:$DATA/ckpt/$RUN/events/" "$DEST/ckpt/$RUN/events/" || echo "no event log yet"
rsync -az --include='iter_*/' --include='iter_*/adapter/' --exclude='*' "$HOST:$DATA/ckpt/$RUN/" "$DEST/ckpt/$RUN/" || true
rsync -az -m "${TRIAL_FILES[@]}" --exclude='*' "$HOST:$DATA/trials/$RUN/" "$DEST/trials/$RUN/" || echo "no training trials yet"
rsync -az -m "${TRIAL_FILES[@]}" --exclude='*' "$HOST:$DATA/trials/eval/$RUN/" "$DEST/trials/eval/$RUN/" 2>/dev/null || true
rsync -az "$HOST:$DATA/monitor/$RUN/host.jsonl" "$DEST/monitor/$RUN/" || true
ssh "$HOST" "curl -sf http://127.0.0.1:8770/status.json" > "$DEST/${RUN}_status.json" \
  || { rm -f "$DEST/${RUN}_status.json"; echo "tracker not running on the box: no status.json snapshot"; }
echo "copied to $DEST/   view: python3 monitor/server.py --run $RUN --data $DEST   or open monitor/index.html and the snapshot"
