#!/usr/bin/env bash
# The GRPO run's sandbox boxes on EC2 (dist_setup.md). Runs on the laptop with its AWS CLI profile.
#
#   bash scripts/ec2_sandboxes.sh up [N]                   # launch N boxes (default 3) and prepare them in parallel
#   bash scripts/ec2_sandboxes.sh allow <h200-ip> [user]   # H200 may ssh in (tunnels, trial sync); copies its keys there
#   bash scripts/ec2_sandboxes.sh serve <run> [t3]         # (re)start each box's agent server; t3: 2-turn trials
#   bash scripts/ec2_sandboxes.sh probe [h200-ip]           # reverse tunnels (during preflight T0); rows -> H200
#   bash scripts/ec2_sandboxes.sh status | down
#
# Boxes terminate themselves after TTL_MIN whatever happens here. State: state/boxes.env, state/box_<i>.log,
# and two keys the H200 uses: tunnel_key (port forwarding only) and sync_key (read-only rsync of ~/data/trials).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
STATE=${STATE:-$HERE/state}
REGION=${REGION:-us-east-2}
TYPE=${TYPE:-m7a.8xlarge}                                # 32 vCPU, 128 GB; spot interruption band 5-10%
MARKET=${MARKET:-spot}                                   # spot, or on-demand
KEY_NAME=${KEY_NAME:-aees-ec2}
KEY=${KEY:-$HOME/.ssh/aees-ec2.pem}
TTL_MIN=${TTL_MIN:-900}                                  # 15 h: setup + tests + 10 h training + margin
SG_NAME=aees-grpo
source "$HERE/scripts/ec2_lib.sh"
mkdir -p "$STATE"
ssh_box() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$1" "${@:2}"; }
boxes() { source "$STATE/boxes.env"; echo "$BOX_IPS"; }

prepare_box() {       # prepare_box <i> <ip>: copy what the box needs, then scripts/ec2_box.sh prepare
  ec2_wait_ssh "$KEY" "$2"
  ssh_box "$2" 'mkdir -p geodesic/aees/post_training/{scripts,configs,grpo,data} geodesic/aees/harbor'
  local rsh="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
  rsync -az -e "$rsh" --exclude .venv --exclude '*.sqlite' --exclude 'state/remote/' --exclude '*.log' \
    --exclude __pycache__ "$HERE/../harbor/" "ubuntu@$2:geodesic/aees/harbor/"
  rsync -az -e "$rsh" "$HERE/setup.sh" "ubuntu@$2:geodesic/aees/post_training/"
  rsync -az -e "$rsh" "$HERE/scripts/ec2_box.sh" "$HERE/scripts/agent_server.sh" "ubuntu@$2:geodesic/aees/post_training/scripts/"
  rsync -az -e "$rsh" "$HERE/configs/pins.sh" "ubuntu@$2:geodesic/aees/post_training/configs/"
  rsync -az -e "$rsh" "$HERE/grpo/harbor_parser.patch" "ubuntu@$2:geodesic/aees/post_training/grpo/"
  rsync -az -e "$rsh" "$STATE/tunnel_key.pub" "$STATE/sync_key.pub" "ubuntu@$2:"
  ssh_box "$2" 'bash geodesic/aees/post_training/scripts/ec2_box.sh prepare'
}

case "${1:-}" in
up)
  N=${2:-3}
  [ ! -f "$STATE/boxes.env" ] || { echo "boxes already up ($(boxes)); run: $0 down"; exit 1; }
  for k in tunnel_key sync_key; do
    [ -f "$STATE/$k" ] || ssh-keygen -q -t ed25519 -N '' -C "aees-grpo-$k" -f "$STATE/$k"
  done
  ec2_identity
  AMI=$(ec2_ami "$REGION")
  SG=$(ec2_sg "$REGION" "$SG_NAME" "aees GRPO sandboxes: SSH from the laptop and the H200 only")
  market=(); [ "$MARKET" = spot ] && market=(--instance-market-options \
    'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  launch() {
    aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY_NAME" \
      --count "$N" --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate "$@" \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
      --user-data "$(printf '#!/bin/bash\nshutdown -h +%s\n' "$TTL_MIN")" \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aees-grpo-box},{Key=Project,Value=aees}]' \
      --query 'Instances[].InstanceId' --output text
  }
  IDS=$(launch "${market[@]}") || { echo "spot launch failed; on-demand instead"; MARKET=on-demand; IDS=$(launch); }
  printf 'REGION=%s\nBOX_IDS="%s"\nSG=%s\n' "$REGION" "$IDS" "$SG" > "$STATE/boxes.env"
  echo "launched $IDS ($N x $TYPE, $MARKET); they terminate themselves after $TTL_MIN min"
  aws ec2 wait instance-running --region "$REGION" --instance-ids $IDS
  IPS=$(aws ec2 describe-instances --region "$REGION" --instance-ids $IDS \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text)
  echo "BOX_IPS=\"$IPS\"" >> "$STATE/boxes.env"
  i=0; pids=()
  for ip in $IPS; do
    i=$((i + 1)); prepare_box "$i" "$ip" > "$STATE/box_$i.log" 2>&1 & pids+=($!)
  done
  fail=0
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}" || { echo "box $((k + 1)) failed: tail $STATE/box_$((k + 1)).log"; tail -5 "$STATE/box_$((k + 1)).log"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "boxes ready: $IPS   next: $0 allow <h200-ip>"
  exit "$fail" ;;
allow)
  H200=${2:?usage: $0 allow <h200-ip> [user]}; HU=${3:-ubuntu}
  source "$STATE/boxes.env"
  ec2_allow_ssh "$REGION" "$SG" "$H200"
  SSH_H=(ssh ${H200_KEY:+-i "$H200_KEY"} -o StrictHostKeyChecking=accept-new "$HU@$H200")
  "${SSH_H[@]}" 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
  scp -q ${H200_KEY:+-i "$H200_KEY"} "$STATE/tunnel_key" "$HU@$H200:.ssh/aees_tunnel"
  scp -q ${H200_KEY:+-i "$H200_KEY"} "$STATE/sync_key" "$HU@$H200:.ssh/aees_sync"
  "${SSH_H[@]}" "chmod 600 ~/.ssh/aees_tunnel ~/.ssh/aees_sync && echo '$BOX_IPS' > ~/aees_boxes"
  echo "H200 $H200 may reach: $BOX_IPS (keys in ~/.ssh/aees_{tunnel,sync}, list in ~/aees_boxes)" ;;
serve)
  RUN=${2:?usage: $0 serve <run> [t3]}; MODE=${3:-normal}
  pids=(); for ip in $(boxes); do ssh_box "$ip" "bash geodesic/aees/post_training/scripts/ec2_box.sh serve $RUN $MODE" & pids+=($!); done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] && echo "agent servers ($MODE) up on: $(boxes)" || echo "some agent servers did not start (see above)"
  exit "$fail" ;;
probe)                # each box: 16 reverse-tunnel listeners, and the H200's T0 stub answering through them
  rows=""; i=0
  for ip in $(boxes); do
    i=$((i + 1))
    n=$(ssh_box "$ip" "ss -ltn | grep -cE '127.0.0.1:300(0[0-9]|1[0-5]) '"); got=$(ssh_box "$ip" 'curl -s --max-time 5 127.0.0.1:30000')
    ok=0; [ "$n" = 16 ] && [ "$got" = aees-stub ] && ok=1
    echo "box $i ($ip): $n/16 reverse listeners, H200 stub: ${got:-no answer}"
    rows+=$(printf '{"t": %s, "test": "T0", "check": "box %s reverse tunnels (%s)", "ok": %s, "value": "%s/16 listeners, stub: %s"}' \
      "$(date +%s)" "$i" "$ip" "$([ $ok = 1 ] && echo true || echo false)" "$n" "${got:-none}")$'\n'
  done
  if [ -n "${2:-}" ]; then
    [ -n "${H200_USER:-}" ] || H200_USER=$(sed -n 's/^H200_USER=//p' "$STATE/run.env" 2>/dev/null || true)   # kickoff.sh's
    printf '%s' "$rows" | ssh ${H200_KEY:+-i "$H200_KEY"} "${H200_USER:-ubuntu}@$2" \
      "cat >> ${DATA:-/data}/monitor/\$(cat ~/aees_run)/preflight.jsonl"
  fi ;;
status)
  for ip in $(boxes); do
    echo "== $ip: $(ssh_box "$ip" 'curl -s 127.0.0.1:11000/health; echo " containers: $(docker ps -q | wc -l)"' 2>&1)"
  done ;;
down)
  [ -f "$STATE/boxes.env" ] || { echo "no state/boxes.env: nothing to terminate"; exit 0; }
  source "$STATE/boxes.env"
  aws ec2 terminate-instances --region "$REGION" --instance-ids $BOX_IDS \
    --query 'TerminatingInstances[].CurrentState.Name' --output text
  rm "$STATE/boxes.env" ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
