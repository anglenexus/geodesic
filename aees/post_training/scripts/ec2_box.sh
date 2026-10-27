#!/usr/bin/env bash
# On one EC2 sandbox box (run by scripts/ec2_sandboxes.sh). Same layout as the H200 (DATA=~/data), same scripts.
#
#   bash ec2_box.sh prepare            # Docker, Harbor's agent server, 20 training-task images, the H200's keys
#   bash ec2_box.sh serve <run> [t3]   # (re)start the agent server in tmux "agent" on 127.0.0.1:11000
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
export DATA=$HOME/data PATH="$HOME/.local/bin:$PATH"
SLOTS=${SLOTS:-32}                                       # trials at once (dist_setup.md: <= ~50 per box)
FWD_PORTS=$(seq -s ' ' 30000 30015)                      # Miles' session servers, reached through the H200's -R

case "${1:-}" in
prepare)
  mkdir -p "$DATA"/{tasks,src,trials,docker}
  DATA=$DATA STEPS="2 3 5 8" TASK_LISTS=tb21_selected_tasks.txt bash "$HERE/setup.sh"   # the 20 training tasks
  command -v rrsync >/dev/null || { echo "rrsync missing (rsync >= 3.2.4 ships it)"; exit 1; }
  # The H200's two keys: one may only forward the agent server and session-server ports, one may only read trials.
  listen=$(for p in $FWD_PORTS; do printf 'permitlisten="127.0.0.1:%s",' "$p"; done)
  touch ~/.ssh/authorized_keys
  grep -qF "$(cut -d' ' -f2 ~/tunnel_key.pub)" ~/.ssh/authorized_keys || \
    echo "restrict,port-forwarding,permitopen=\"127.0.0.1:11000\",${listen%,} $(cat ~/tunnel_key.pub)" >> ~/.ssh/authorized_keys
  grep -qF "$(cut -d' ' -f2 ~/sync_key.pub)" ~/.ssh/authorized_keys || \
    echo "restrict,command=\"rrsync -ro $DATA/trials\" $(cat ~/sync_key.pub)" >> ~/.ssh/authorized_keys
  sg docker -c "docker images --format '{{.Repository}}' | grep -c . | sed 's/^/task images: /'"
  echo "prepared: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM, disk free $(df -h ~ | tail -1 | awk '{print $4}')" ;;
serve)
  RUN=${2:?usage: ec2_box.sh serve <run> [t3]}
  t3=""; [ "${3:-}" = t3 ] && t3="HARBOR_AGENT_MAX_ITERATIONS=2 HARBOR_VERIFIER_TIMEOUT_SEC=60"   # integration test T3
  tmux kill-session -t agent 2>/dev/null || true
  while curl -sf 127.0.0.1:11000/health >/dev/null 2>&1; do sleep 1; done     # the old server is gone
  tmux new -d -s agent "sg docker -c 'env $t3 DATA=$DATA RUN=$RUN CONCURRENCY=$SLOTS bash $HERE/scripts/agent_server.sh'"
  for _ in $(seq 120); do curl -sf 127.0.0.1:11000/health >/dev/null 2>&1 && { echo "agent server up ($RUN ${3:-normal})"; exit 0; }; sleep 1; done
  echo "agent server did not come up:"; tail -20 "$DATA/trials/$RUN/agent_server.log" 2>/dev/null; exit 1 ;;
*) sed -n 2,5p "$0"; exit 2 ;;
esac
