#!/usr/bin/env bash
# From the laptop, once the H200's IP is known (dist_setup.md): everything up to "ready for tests", in parallel.
#
#   bash scripts/kickoff.sh <h200-ip> [h200-user]       # H200_KEY=~/.ssh/<key> if not the default key
#
# 1. probe the H200 (driver >= 580, else stop: dist_setup.md H200 item 1)   2. launch + prepare 3 EC2 boxes (background)
# 3. code and prompt files -> H200; scripts/h200_setup.sh (steps 1-4 over ssh, the rest in tmux "setup")
# 4. boxes: allow the H200, agent servers for $RUN   5. H200: links (tunnels, sync, dispatcher) and the tracker.
# Then, after SETUP_DONE, scripts/launch.sh on the H200. The integration tests (integration_tests.md) passed on
# 2026-10-27 and aren't re-run; a new failure goes into integration_issues.md.
set -euo pipefail
H200=${1:?usage: kickoff.sh <h200-ip> [h200-user]}
HU=${2:-ubuntu}
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
STATE=${STATE:-$HERE/state}
mkdir -p "$STATE"
STAMP=$(date +%m%d_%H%M)
# A previous run's state would be reused: its run names (new H200) and its boxes (terminated by finish.sh).
old=""
if [ -f "$STATE/run.env" ] && ! grep -qx "H200=$H200" "$STATE/run.env"; then old="run.env"; fi
if [ -f "$STATE/boxes.env" ]; then
  ids=$(source "$STATE/boxes.env"; echo $BOX_IDS)
  live=$(aws ec2 describe-instances --region "${REGION:-us-east-2}" --instance-ids $ids \
    --filters Name=instance-state-name,Values=pending,running --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 0)
  [ "${live:-0}" -gt 0 ] || old="$old boxes.env"
fi
if [ -n "$old" ]; then
  mkdir -p "$STATE/old_$STAMP"
  for f in $old kickoff.log up.log box_1.log box_2.log box_3.log; do [ ! -f "$STATE/$f" ] || mv "$STATE/$f" "$STATE/old_$STAMP/"; done
  echo "previous run's state moved to state/old_$STAMP"
fi
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
export COPYFILE_DISABLE=1                                # no macOS metadata in the archives (GNU tar warns on it)
tar --no-xattrs -C "$HERE" --exclude ./.venv --exclude ./data --exclude ./state --exclude __pycache__ --exclude .pytest_cache \
  --exclude ./proto/state -czf - . | "${SSH_H[@]}" 'mkdir -p geodesic/aees/post_training && tar -xzf - -C geodesic/aees/post_training'
tar --no-xattrs -C "$HERE/.." -czf - harbor/state/tb21_selected_tasks.txt harbor/state/tb21_val8_tasks.txt harbor/state/tb21_test20_tasks.txt \
  harbor/scripts/task_images.py | "${SSH_H[@]}" 'tar -xzf - -C geodesic/aees'
tar --no-xattrs -C "$HERE" -czf - data/grpo_train.jsonl data/grpo_val.jsonl | "${SSH_H[@]}" 'tar -xzf - -C geodesic/aees/post_training'
"${SSH_H[@]}" 'PHASE=pre bash ~/geodesic/aees/post_training/scripts/h200_setup.sh > ~/setup.log 2>&1' \
  || { "${SSH_H[@]}" tail -20 setup.log; exit 1; }       # before any tmux: sessions then get the docker group
"${SSH_H[@]}" "tmux kill-session -t setup 2>/dev/null; tmux new -d -s setup 'PHASE=main bash ~/geodesic/aees/post_training/scripts/h200_setup.sh 2>&1 | tee -a ~/setup.log'"

el "4/5 waiting for the boxes"
if [ -f "$STATE/up.pid" ]; then
  wait "$(cat "$STATE/up.pid")" 2>/dev/null || true
  while kill -0 "$(cat "$STATE/up.pid")" 2>/dev/null; do sleep 5; done
  rm -f "$STATE/up.pid"
fi
tail -3 "$STATE/up.log"
[ -f "$STATE/boxes.env" ] || { echo "boxes failed: see state/up.log"; exit 1; }
bash "$HERE/scripts/ec2_sandboxes.sh" allow "$H200" "$HU"
(source "$STATE/boxes.env"; echo $BOX_IDS) | "${SSH_H[@]}" 'cat > ~/aees_box_ids'   # finish.sh terminates these
bash "$HERE/scripts/ec2_sandboxes.sh" serve "$RUN"

el "5/5 H200: links + tracker for $RUN"
"${SSH_H[@]}" "cd geodesic/aees/post_training && RUN=$RUN SLOTS=32 bash scripts/links.sh start && \
  (tmux kill-session -t tracker 2>/dev/null; tmux new -d -s tracker 'RUN=$RUN python3 monitor/server.py')"
# The dashboard on this laptop: a background tunnel to the H200's tracker (replaces an older one on the port).
pkill -f "ssh .*-L 8770:127.0.0.1:8770" 2>/dev/null || true
lsof -ti tcp:8770 -sTCP:LISTEN 2>/dev/null | xargs -r kill 2>/dev/null || true
ssh ${H200_KEY:+-i "$H200_KEY"} -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 8770:127.0.0.1:8770 "$HU@$H200" \
  && { command -v open >/dev/null && open http://127.0.0.1:8770/ || true; }
el "ready. dashboard: http://127.0.0.1:8770/   setup on the H200: ${SSH_H[*]} tail -3 setup.log   (wait for SETUP_DONE)"
cat <<NEXT
Next, on the H200 (${SSH_H[*]}), once: aws configure; nebius profile create (summary.md, step 3). Then in
geodesic/aees/post_training, after SETUP_DONE:
  RUN=$RUN STEPS=<n> [INIT_ADAPTER=/root/ckpt/<run>/iter_<N>/adapter] NEBIUS_INSTANCE_ID=<id> bash scripts/launch.sh
Dashboard: http://127.0.0.1:8770/ (tunnel opened above; reopen with: bash scripts/tunnel.sh $HU@$H200)
NEXT
