#!/usr/bin/env bash
# From the laptop: Experiment 1 of align_tmax_plan.md up to "ready to start". Creates the 8xH100 host (Nebius
# eu-north1, on demand), launches 2 on-demand EC2 boxes in phase-1 mode (20 slots, 65k context, 64 model calls,
# 120 s per command), and sets the host up for serving only (Miles image + weights; no torch_dist, no Q adapter).
#
#   bash scripts/phase1_kickoff.sh            # then follow the printed steps
# Spend: ~$36/h for the host + ~$3.70/h for the boxes from the moment they exist. Raise the budget guard first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
STATE=${STATE:-$HERE/state}
STAMP=$(date +%m%d_%H%M)
RUN=${RUN:-phase1_$STAMP}
[ ! -f "$STATE/nebius.env" ] || { echo "state/nebius.env exists (a host from an earlier run?): move it away first"; exit 1; }
PROJECT=${PROJECT:-project-e00qzjr2pr00d8qnh3d9xa} PLATFORM=gpu-h100-sxm PRESET=${PRESET:-8gpu-128vcpu-1600gb} \
  NAME=${NAME:-aees-h100x8-$(date +%m%d)} DISK_GB=${DISK_GB:-512} bash "$HERE/scripts/nebius_up.sh"
source "$STATE/nebius.env"
MARKET=on-demand TTL_MIN=${TTL_MIN:-300} RUN_NAME=$RUN SERVE_MODE=phase1 SERVE_ONLY=1 NO_TRACKER=1 \
  bash "$HERE/scripts/kickoff.sh" "$H200_IP" "$H200_USER"
cat <<NEXT
Host $NEBIUS_INSTANCE_ID ($H200_IP) and boxes are up; run $RUN. Next:
  1. laptop: bash scripts/teardown_watch.sh $RUN $NEBIUS_INSTANCE_ID &      (deletes the host on BACKUP_OK)
  2. host (ssh $H200_USER@$H200_IP), once: aws configure. Wait for SETUP_DONE in ~/setup.log, then in
     geodesic/aees/post_training:   RUN=$RUN NEBIUS_INSTANCE_ID=$NEBIUS_INSTANCE_ID bash scripts/phase1.sh
  3. laptop: PORT=8771 bash scripts/tunnel.sh $H200_USER@$H200_IP     ->  http://localhost:8771
NEXT
