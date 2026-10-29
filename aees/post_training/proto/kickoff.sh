#!/usr/bin/env bash
# Start the overlap prototype (PLAN.md §5b) from the laptop, after proto/ec2_up.sh:
#
#   bash proto/kickoff.sh <h200-ip> [h200-user]       # H200_KEY=~/.ssh/<key> if not the default key
#
# 1. copies aees/post_training to the H200; 2. lets the EC2 box tunnel to the H200's SGLang port (a key that
# can only forward 127.0.0.1:30000); 3. on the H200, in tmux "proto": setup.sh (steps 1-4, 6, 7) then
# proto/bench.sh (~25 min setup + ~50 min phases); 4. on EC2, in tmux "proto": proto/ec2_trials.sh.
# Afterwards: bash proto/fetch.sh (results + report), then bash proto/ec2_down.sh.
set -euo pipefail
H200=${1:?usage: kickoff.sh <h200-ip> [h200-user]}
HU=${2:-ubuntu}
HERE=$(cd "$(dirname "$0")/.." && pwd)                    # aees/post_training
STATE=$HERE/proto/state
source "$STATE/ec2.env"                                   # EC2_IP (from ec2_up.sh)
KEY=${KEY:-$HOME/.ssh/aees-ec2.pem}
RUN=${RUN:-$(date +%Y%m%d_%H%M)}
printf 'RUN=%s\nH200=%s\nH200_USER=%s\n' "$RUN" "$H200" "$HU" > "$STATE/run.env"
SSH_H=(ssh ${H200_KEY:+-i "$H200_KEY"} -o StrictHostKeyChecking=accept-new "$HU@$H200")
SSH_E=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$EC2_IP")

echo "== 1/4 code -> H200"
tar -C "$HERE" --exclude ./.venv --exclude ./data --exclude __pycache__ --exclude .pytest_cache --exclude ./proto/state \
  -czf - . | "${SSH_H[@]}" 'mkdir -p geodesic/aees/post_training && tar -xzf - -C geodesic/aees/post_training'   # no rsync yet

echo "== 2/4 tunnel key: EC2 may forward to the H200's 127.0.0.1:30000, nothing else"
[ -f "$STATE/tunnel_key" ] || ssh-keygen -q -t ed25519 -N '' -C aees-proto-tunnel -f "$STATE/tunnel_key"
pub=$(cat "$STATE/tunnel_key.pub")
"${SSH_H[@]}" "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && grep -qF '${pub#* }' ~/.ssh/authorized_keys \
  || echo 'restrict,port-forwarding,permitopen=\"127.0.0.1:30000\" $pub' >> ~/.ssh/authorized_keys"
scp -q -i "$KEY" "$STATE/tunnel_key" "ubuntu@$EC2_IP:.ssh/proto_tunnel"
"${SSH_E[@]}" chmod 600 .ssh/proto_tunnel

echo "== 3/4 H200: setup + bench in tmux 'proto' (log ~/proto_$RUN.log)"
"${SSH_H[@]}" "cat > ~/proto_run.sh" <<RUNSH
cd ~/geodesic/aees/post_training
STEPS="1 2 3 4 6 7" MODELS="Qwen/Qwen3.6-27B Qwen/Qwen3.6-27B-FP8" bash setup.sh \
  && sg docker -c "RUN=$RUN bash proto/bench.sh"
RUNSH
"${SSH_H[@]}" "tmux new -d -s proto 'bash ~/proto_run.sh 2>&1 | tee -a ~/proto_$RUN.log'"

echo "== 4/4 EC2: real trials in tmux 'proto' (log ~/proto/trials.log)"
"${SSH_E[@]}" "tmux new -d -s proto 'H200=$H200 H200_USER=$HU RUN=$RUN bash ~/proto/ec2_trials.sh 2>&1 | tee -a ~/proto/trials.log'"
cat <<NEXT
Started run $RUN.
  H200:  ${SSH_H[*]} tail -f proto_$RUN.log          (phases: /data/proto/$RUN/bench.log)
  EC2:   ${SSH_E[*]} tail -f proto/trials.log
  Done when bench.log says "done". Then: bash proto/fetch.sh && bash proto/ec2_down.sh
NEXT
