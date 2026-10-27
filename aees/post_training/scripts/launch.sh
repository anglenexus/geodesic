#!/usr/bin/env bash
# On the H200, after kickoff.sh and SETUP_DONE: start a GRPO run with its watchers (summary.md, step 5).
#   tmux links     tunnels, trial sync, dispatcher (SLOTS per box)
#   tmux tracker   dashboard on 127.0.0.1:8770
#   tmux train     scripts/run.sh grpo, a checkpoint every step
#   tmux deadline  stops train after DEADLINE_H hours
#   tmux hourly    S3 backup every hour while train runs
#   tmux finish    when train ends: backup, size check, terminate the boxes, delete this H200 (finish.sh)
#
#   RUN=grpoc_1027_2058 STEPS=4 INIT_ADAPTER=/root/ckpt/grpo10h_r2/iter_0000004/adapter \
#     PRIOR_SNAPSHOT=/data/monitor/grpo10h_r2_status.json NEBIUS_INSTANCE_ID=computeinstance-... bash scripts/launch.sh
# PRIOR_SNAPSHOT: the earlier run's final dashboard (S3 <run>/monitor/dashboard/<run>_status.json); the tracker
# shows its trained steps first and numbers this run's steps after them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
export DATA=${DATA:-/data}
RUN=${RUN:?set RUN}
STEPS=${STEPS:-20}                                       # rollout + training steps (Miles --num-rollout)
SLOTS=${SLOTS:-32}                                       # per box: 2 x 32 = 64, the concurrency chosen in T2 (ec2_sizing.md)
DEADLINE_H=${DEADLINE_H:-10}
DEST=${DEST:-s3://aees-grpo-backups-611728564241}
INIT_ADAPTER=${INIT_ADAPTER:-}                           # container path; weights only, fresh Adam (summary.md, step 4)
[ -z "$INIT_ADAPTER" ] || [ -f "$DATA/ckpt/${INIT_ADAPTER#/root/ckpt/}/adapter_megatron_rank0.pt" ] \
  || { echo "no adapter at $DATA/ckpt/${INIT_ADAPTER#/root/ckpt/}"; exit 1; }
for s in train deadline hourly finish; do
  tmux has-session -t "$s" 2>/dev/null && { echo "tmux $s already running: stop it first"; exit 1; }
done
aws sts get-caller-identity >/dev/null 2>&1 || echo "WARNING: no AWS credentials (aws configure): no backups, no teardown"
[ -z "${NEBIUS_INSTANCE_ID:-}" ] || nebius compute instance get --id "$NEBIUS_INSTANCE_ID" >/dev/null 2>&1 \
  || echo "WARNING: nebius CLI can't see $NEBIUS_INSTANCE_ID (nebius profile create): the H200 won't be deleted"

mkdir -p "$DATA/monitor/$RUN"
sudo docker ps -q | xargs -r sudo docker rm -f >/dev/null   # leftovers of an earlier run
cd "$HERE"
RUN=$RUN SLOTS=$SLOTS bash scripts/links.sh start | tail -1 | cut -c1-120; echo
PRIOR_SNAPSHOT=${PRIOR_SNAPSHOT:-}
prior=0
[ -z "$PRIOR_SNAPSHOT" ] || prior=$(python3 -c 'import json,sys; print(sum(any(k.startswith("train/") for k in r) for r in json.load(open(sys.argv[1]))["steps"]))' "$PRIOR_SNAPSHOT")
tmux kill-session -t tracker 2>/dev/null || true
tmux new -d -s tracker "RUN=$RUN NUM_ROLLOUT=$((prior + STEPS)) PRIOR_SNAPSHOT=$PRIOR_SNAPSHOT python3 monitor/server.py"
tmux new -d -s train "SAVE_DIR=/root/ckpt/$RUN NUM_ROLLOUT=$STEPS SAVE_INTERVAL=1 INIT_ADAPTER=$INIT_ADAPTER \
  bash scripts/run.sh grpo > $DATA/monitor/$RUN/train_run.log 2>&1"
tmux new -d -s deadline "sleep $((DEADLINE_H * 3600)); tmux kill-session -t train; sudo docker ps -q | xargs -r sudo docker rm -f; \
  echo STOPPED_AT_DEADLINE >> $DATA/monitor/$RUN/train_run.log"
tmux new -d -s hourly "while tmux has-session -t train 2>/dev/null; do sleep 3600; sudo chown -R \$USER $DATA/ckpt/$RUN && \
  WITH_OPTIMIZER=1 bash scripts/backup.sh $DATA/ckpt/$RUN $DEST 2>&1 | tail -1 >> $DATA/monitor/hourly_backup.log; done"
tmux new -d -s finish "RUN=$RUN DEST=$DEST WAIT_FOR=train NEBIUS_INSTANCE_ID=${NEBIUS_INSTANCE_ID:-} \
  bash scripts/finish.sh > $DATA/monitor/finish.log 2>&1"
echo "started $RUN at $(date -u +%T) UTC: $STEPS steps, deadline ${DEADLINE_H} h; log $DATA/monitor/$RUN/train_run.log"
