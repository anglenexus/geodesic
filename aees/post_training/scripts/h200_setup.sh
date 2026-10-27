#!/usr/bin/env bash
# On the H200: setup.sh with the slow steps in parallel (dist_setup.md: <= 10 min to "ready for tests").
#   steps 1-4 (checks, packages, Docker config) -> Miles image + source (6) || weights (7) -> torch_dist (9)
# Harbor and task images (5, 8) aren't needed here: the agent servers run on the EC2 boxes.
# Times go to $DATA/monitor/setup_times.txt (copy them into dist_setup.md).
#   PHASE=pre   folders, firewall, steps 1-4, AWS CLI (~35 s; kickoff.sh runs it over plain ssh, before any tmux,
#               so tmux sessions get the docker group that step 2 adds)
#   PHASE=main  steps 6 || 7, then 9 (kickoff.sh runs it in tmux "setup")       default: both
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
export DATA=${DATA:-/data} MODELS=${MODELS:-Qwen/Qwen3.6-27B}
PHASE=${PHASE:-all}
T0=$(date +%s)
stamp() { echo "$(( $(date +%s) - T0 )) s  $1" | tee -a "$DATA/monitor/setup_times.txt"; }
if [ "$PHASE" != main ]; then
sudo mkdir -p "$DATA"/{monitor,cache,proto,ckpt,trials} && sudo chown "$USER" "$DATA" "$DATA"/{monitor,cache,proto,ckpt,trials}
echo "== setup $(date -u +%FT%TZ)" >> "$DATA/monitor/setup_times.txt"
# Miles binds its session servers, router and SGLang to the VM's address: drop new inbound TCP on the public
# interface except ssh (local traffic goes over lo; the EC2 tunnels are outbound ssh).
IFACE=$(ip route get 1.1.1.1 | grep -oP 'dev \K\S+')
sudo iptables -C INPUT -i "$IFACE" -p tcp ! --dport 22 -m conntrack --ctstate NEW -j DROP 2>/dev/null \
  || sudo iptables -I INPUT 1 -i "$IFACE" -p tcp ! --dport 22 -m conntrack --ctstate NEW -j DROP

STEPS="1 2 3 4" bash "$HERE/setup.sh" > "$DATA/monitor/setup_1-4.log" 2>&1 || { tail -20 "$DATA/monitor/setup_1-4.log"; exit 1; }
stamp "steps 1-4 (checks, packages, Docker, container toolkit)"
# finish.sh backs up to S3 and terminates the boxes with the AWS CLI; the Nebius image ships `nebius` but not `aws`.
command -v aws >/dev/null || sudo snap install aws-cli --classic >/dev/null
[ "$PHASE" = pre ] && { echo "PRE_DONE"; exit 0; }
fi
sg docker -c "STEPS=6 bash $HERE/setup.sh" > "$DATA/monitor/setup_6.log" 2>&1 & img=$!
STEPS=7 bash "$HERE/setup.sh" > "$DATA/monitor/setup_7.log" 2>&1 & wts=$!
wait "$wts" || { tail -20 "$DATA/monitor/setup_7.log"; exit 1; }; stamp "weights ($MODELS)"
wait "$img" || { tail -20 "$DATA/monitor/setup_6.log"; exit 1; }; stamp "Miles image + source"
echo "READY_FOR_T0"                                      # T0 can run while the conversion does
sg docker -c "STEPS=9 bash $HERE/setup.sh" > "$DATA/monitor/setup_9.log" 2>&1 || { tail -20 "$DATA/monitor/setup_9.log"; exit 1; }
stamp "torch_dist conversion"
echo "SETUP_DONE"
