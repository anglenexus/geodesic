#!/usr/bin/env bash
# On the H200: the links to the EC2 sandbox boxes (dist_setup.md), in tmux "links":
#   tunnel<i>  one ssh per box: -L 1100<i> -> the box's agent server; -R 30000-30015 -> Miles' session servers here
#   sync       every 60 s: the boxes' trials -> $DATA/trials/$RUN (the tracker reads them there), agent-server logs
#              -> $DATA/monitor/$RUN/box_<i>.log
#   dispatch   grpo/dispatch.py on 127.0.0.1:11000 (Miles' AGENT_SERVER_URL): longest task first, SLOTS per box
#
#   RUN=grpo_10h bash scripts/links.sh start | stop | status
# Needs ~/aees_boxes and ~/.ssh/aees_{tunnel,sync} (laptop: scripts/ec2_sandboxes.sh allow <h200-ip>).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
DATA=${DATA:-/data}
RUN=${RUN:?set RUN, e.g. RUN=grpo_10h}
SLOTS=${SLOTS:-32}
BOXES_FILE=${BOXES_FILE:-$HOME/aees_boxes}
read -ra BOXES < "$BOXES_FILE"
SSHO="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes"

tunnel_cmd() {        # tunnel_cmd <i> <ip>: a loop that keeps one box's forwards up
  local fwd="-L 1100$1:127.0.0.1:11000" p
  for p in $(seq 30000 30015); do fwd+=" -R 127.0.0.1:$p:127.0.0.1:$p"; done
  echo "while true; do ssh -N -i ~/.ssh/aees_tunnel $SSHO -o ExitOnForwardFailure=yes $fwd ubuntu@$2;" \
       "echo \"\$(date +%T) tunnel $2 down, retrying\"; sleep 3; done"
}

sync_cmd() {          # a loop that copies every box's trials (read-only key: rrsync rooted at the box's ~/data/trials)
  local cmds="" i=0 ip
  for ip in "${BOXES[@]}"; do
    i=$((i + 1))
    cmds+="rsync -a --exclude /agent_server.log -e 'ssh -i ~/.ssh/aees_sync $SSHO' ubuntu@$ip:$RUN/ $DATA/trials/$RUN/;"
    cmds+="rsync -a -e 'ssh -i ~/.ssh/aees_sync $SSHO' ubuntu@$ip:$RUN/agent_server.log $DATA/monitor/$RUN/box_$i.log;"
  done
  echo "while true; do $cmds sleep ${SYNC_SEC:-60}; done"
}

case "${1:-}" in
start)
  mkdir -p "$DATA/trials/$RUN" "$DATA/monitor/$RUN"
  echo "$RUN" > ~/aees_run                              # where the laptop's probe appends its rows
  tmux kill-session -t links 2>/dev/null || true
  backends=""
  for i in "${!BOXES[@]}"; do
    n=$((i + 1)); backends+="127.0.0.1:1100$n,"
    if [ "$n" = 1 ]; then tmux new -d -s links -n tunnel1 "$(tunnel_cmd 1 "${BOXES[$i]}")"
    else tmux new-window -t links -n "tunnel$n" "$(tunnel_cmd "$n" "${BOXES[$i]}")"; fi
  done
  tmux new-window -t links -n sync "$(sync_cmd) 2>&1 | grep -v '^$' >> $DATA/monitor/$RUN/sync.log"
  tmux new-window -t links -n dispatch "python3 $HERE/grpo/dispatch.py --backends ${backends%,} --slots $SLOTS \
    --minutes $HERE/grpo/task_minutes.json --log $DATA/trials/$RUN/dispatch.jsonl 2>&1 | tee -a $DATA/monitor/$RUN/dispatch.log"
  for _ in $(seq 30); do curl -sf 127.0.0.1:11000/health >/dev/null 2>&1 && break; sleep 1; done
  bash "$0" status ;;
status)
  tmux list-windows -t links -F '#W' 2>/dev/null | tr '\n' ' ' | sed 's/^/windows: /'; echo
  curl -sf 127.0.0.1:11000/health || echo "dispatcher: not answering"; echo ;;
stop)
  tmux kill-session -t links 2>/dev/null || true; echo "links stopped" ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
