#!/usr/bin/env bash
# From the laptop, once the H200's IP is known (dist_setup.md): everything up to "ready for tests", in parallel.
#
#   bash scripts/kickoff.sh <h200-ip> [h200-user]       # H200_KEY=~/.ssh/<key> if not the default key
#
# 1. probe the H200 (driver >= 580, else stop: dist_setup.md H200 item 1)   2. launch + prepare 3 EC2 boxes (background)
# 3. code and prompt files -> H200; scripts/h200_setup.sh in tmux "setup"   4. boxes: allow the H200, agent servers in
# T3 mode   5. H200: links (tunnels, sync, dispatcher) and the tracker for the T3 run. Then integration_tests.md.
set -euo pipefail
H200=${1:?usage: kickoff.sh <h200-ip> [h200-user]}
HU=${2:-ubuntu}
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
STATE=${STATE:-$HERE/state}
mkdir -p "$STATE"
STAMP=$(date +%m%d_%H%M)
[ -f "$STATE/run.env" ] || printf 'RUN_T3=t3_%s\nRUN=grpo10h_%s\nH200=%s\nH200_USER=%s\n' "$STAMP" "$STAMP" "$H200" "$HU" > "$STATE/run.env"
source "$STATE/run.env"
SSH_H=(ssh ${H200_KEY:+-i "$H200_KEY"} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -x "$HU@$H200")
T0=$(date +%s); el() { echo "[$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s] $*"; }

el "1/5 probe"
drv=$("${SSH_H[@]}" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1')
el "    driver $drv, $("${SSH_H[@]}" 'nproc') vCPU, $("${SSH_H[@]}" "df -h / | tail -1 | awk '{print \$4}'") free"
[ "${drv%%.*}" -ge 580 ] || { echo "driver < 580: see dist_setup.md (H200 item 1), then re-run"; exit 1; }

el "2/5 EC2 boxes (background, log state/up.log)"
[ -f "$STATE/boxes.env" ] || { bash "$HERE/scripts/ec2_sandboxes.sh" up > "$STATE/up.log" 2>&1 & echo $! > "$STATE/up.pid"; }

el "3/5 code -> H200, setup in tmux 'setup' (log ~/setup.log)"
tar -C "$HERE" --exclude ./.venv --exclude ./data --exclude ./state --exclude __pycache__ --exclude .pytest_cache \
  --exclude ./proto/state -czf - . | "${SSH_H[@]}" 'mkdir -p geodesic/aees/post_training && tar -xzf - -C geodesic/aees/post_training'
tar -C "$HERE/.." -czf - harbor/state/tb21_selected_tasks.txt harbor/state/tb21_val8_tasks.txt harbor/state/tb21_test20_tasks.txt \
  harbor/scripts/task_images.py | "${SSH_H[@]}" 'tar -xzf - -C geodesic/aees'
tar -C "$HERE" -czf - data/grpo_train.jsonl data/grpo_val.jsonl | "${SSH_H[@]}" 'tar -xzf - -C geodesic/aees/post_training'
"${SSH_H[@]}" "tmux kill-session -t setup 2>/dev/null; tmux new -d -s setup 'bash ~/geodesic/aees/post_training/scripts/h200_setup.sh 2>&1 | tee ~/setup.log'"

el "4/5 waiting for the boxes"
if [ -f "$STATE/up.pid" ]; then
  wait "$(cat "$STATE/up.pid")" 2>/dev/null || true
  while kill -0 "$(cat "$STATE/up.pid")" 2>/dev/null; do sleep 5; done
  rm -f "$STATE/up.pid"
fi
tail -3 "$STATE/up.log"
[ -f "$STATE/boxes.env" ] || { echo "boxes failed: see state/up.log"; exit 1; }
bash "$HERE/scripts/ec2_sandboxes.sh" allow "$H200" "$HU"
bash "$HERE/scripts/ec2_sandboxes.sh" serve "$RUN_T3" t3

el "5/5 H200: links + tracker for $RUN_T3"
"${SSH_H[@]}" "cd geodesic/aees/post_training && RUN=$RUN_T3 bash scripts/links.sh start && \
  (tmux kill-session -t tracker 2>/dev/null; tmux new -d -s tracker 'RUN=$RUN_T3 NUM_ROLLOUT=2 python3 monitor/server.py')"
el "ready. setup on the H200: ${SSH_H[*]} tail -3 setup.log   (READY_FOR_T0, then SETUP_DONE)"
cat <<NEXT
Next (integration_tests.md), on the H200 in geodesic/aees/post_training:
  RUN=$RUN_T3 bash scripts/preflight.sh t0      laptop meanwhile: bash scripts/ec2_sandboxes.sh probe $H200
  RUN=$RUN_T3 bash scripts/preflight.sh t2      after SETUP_DONE (GPU free)
  RUN=$RUN_T3 SGLANG_EXTRA='<from T2>' bash scripts/preflight.sh t3
Tracker: bash scripts/tunnel.sh $HU@$H200   then http://localhost:8770
NEXT
