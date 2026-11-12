#!/usr/bin/env bash
# On the 8xH100 host, after kickoff (scripts/phase1_kickoff.sh) and SETUP_DONE: Experiment 1 of align_tmax_plan.md,
# 160 rollouts (20 training tasks x 8) of the base model with thinking on, 40 at a time, no training.
#   engines    scripts/phase1_serve.sh: 4 SGLang engines x TP2 on 127.0.0.1:30001-30004
#   tmux links       tunnels, trial sync, dispatcher (20 slots per box, longest time limit first, fixed order)
#   then box_check.sh: every box answers and outlives the run, or nothing starts
#   tmux p1tracker   monitor/phase1.py on 127.0.0.1:8771 (laptop: PORT=8771 bash scripts/tunnel.sh <user>@<ip>)
#   tmux rollout     grpo/rollout_only.py -> $DATA/trials/$RUN/rollouts.jsonl
#   tmux deadline    stops the driver after DEADLINE_MIN minutes
#   tmux finish      when the driver ends: backup to S3, terminate the boxes, delete this host (finish.sh)
#
#   RUN=phase1_1112_0900 NEBIUS_INSTANCE_ID=computeinstance-... bash scripts/phase1.sh
#   FULL=1: Tmax's 88 TB 2.1 tasks (paper §H: all 89 but configure-git-webserver) x 5, Tmax paper reference; `touch $DATA/trials/$RUN/extend` before the 5th attempts
#   finish to run 8 attempts per task instead (same run, deadline already covers it)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
export DATA=${DATA:-/data}
RUN=${RUN:?set RUN}
FULL=${FULL:-}
ATTEMPTS=${ATTEMPTS:-$([ -n "$FULL" ] && echo 5 || echo 8)}
EXTEND_TO=${EXTEND_TO:-$([ -n "$FULL" ] && echo 8 || echo 0)}
SLOTS=${SLOTS:-20}                                       # per box: 2 x 20 = 40 at once
DEADLINE_MIN=${DEADLINE_MIN:-$([ -n "$FULL" ] && echo 330 || echo 100)}   # 20 tasks: median 57 min, p90 59 (align_tmax_plan.md); full: 200-min task + 8 attempts
DEST=${DEST:-s3://aees-grpo-backups-611728564241}
TASKS=${TASKS:-$HERE/../harbor/state/$([ -n "$FULL" ] && echo tb21_tmax88_tasks.txt || echo tb21_selected_tasks.txt)}
LIMITS=${LIMITS:-$HERE/grpo/$([ -n "$FULL" ] && echo task_limits_tb21.json || echo task_limits.json)}
REFERENCE=${REFERENCE:-$HERE/monitor/$([ -n "$FULL" ] && echo tb21_reference.json || echo phase1_reference.json)}
ENGINES=${ENGINES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}                          # Tmax's eval agent (their RL rollouts use 1.0)
TOP_P=${TOP_P:-0.95}
for s in rollout deadline finish; do
  tmux has-session -t "$s" 2>/dev/null && { echo "tmux $s already running: stop it first"; exit 1; }
done
aws sts get-caller-identity >/dev/null 2>&1 || echo "WARNING: no AWS credentials (aws configure): no backup, no teardown"
mkdir -p "$DATA/monitor/$RUN" "$DATA/trials/$RUN"
cd "$HERE"

engines_now=$(RUN=$RUN ENGINES=$ENGINES bash scripts/phase1_serve.sh status)   # not `| grep -q`: pipefail + SIGPIPE skipped start
[[ $engines_now != *down* ]] || RUN=$RUN ENGINES=$ENGINES bash scripts/phase1_serve.sh start
ports=$(seq 30001 $((30000 + ENGINES)) | paste -sd, -)
[ -f "$DATA/monitor/$RUN/probe.json" ] || python3 scripts/phase1_probe.py --ports "$ports" --out "$DATA/monitor/$RUN/probe.json" \
  || echo "probe: thinking check failed, see $DATA/monitor/$RUN/probe.json"   # reported, not a gate
RUN=$RUN SLOTS=$SLOTS MINUTES=$LIMITS DISPATCH_ARGS=--fixed-order bash scripts/links.sh start | tail -1 | cut -c1-120; echo
NEED_MIN=$((DEADLINE_MIN + 30)) bash scripts/box_check.sh \
  || { bash scripts/links.sh stop; echo "box check failed: not starting $RUN"; exit 1; }
EXT_ARGS=$([ "$EXTEND_TO" -gt "$ATTEMPTS" ] && echo "--extend-to $EXTEND_TO --extend-file $DATA/trials/$RUN/extend" || true)
engines=$(for i in $(seq 1 "$ENGINES"); do printf 'http://127.0.0.1:%s/v1,' $((30000 + i)); done)
tmux kill-session -t p1tracker 2>/dev/null || true
printf '{"tasks": %s, "attempts": %s, "extend_attempts": %s, "temperature": %s, "top_p": %s, "context": 65536, "max_tokens": 16384, "max_calls": 64, "command_timeout_sec": 120, "thinking": true}\n' \
  "$(grep -c . "$TASKS")" "$ATTEMPTS" "$EXTEND_TO" "$TEMPERATURE" "$TOP_P" > "$DATA/monitor/$RUN/settings.json"
tmux new -d -s p1tracker "RUN=$RUN python3 monitor/phase1.py --port 8771 --engine-ports $ports --reference $REFERENCE --planned $((ATTEMPTS * $(grep -c . "$TASKS")))"
tmux new -d -s rollout "python3 grpo/rollout_only.py --engines ${engines%,} --tasks $TASKS --attempts $ATTEMPTS \
  --minutes $LIMITS $EXT_ARGS --temperature $TEMPERATURE --top-p $TOP_P --out $DATA/trials/$RUN/rollouts.jsonl > $DATA/monitor/$RUN/rollout.log 2>&1"
tmux new -d -s deadline "sleep $((DEADLINE_MIN * 60)); tmux kill-session -t rollout; echo STOPPED_AT_DEADLINE >> $DATA/monitor/$RUN/rollout.log"
tmux new -d -s finish "RUN=$RUN DEST=$DEST WAIT_FOR=rollout TRACKER_PORT=8771 DASHBOARD_HTML=$HERE/monitor/phase1.html \
  NEBIUS_INSTANCE_ID=${NEBIUS_INSTANCE_ID:-} bash scripts/finish.sh > $DATA/monitor/finish.log 2>&1"
echo "started $RUN at $(date -u +%T) UTC: $((ATTEMPTS * $(grep -c . "$TASKS"))) trials, deadline ${DEADLINE_MIN} min; log $DATA/monitor/$RUN/rollout.log"
