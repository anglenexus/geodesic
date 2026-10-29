#!/usr/bin/env bash
# EC2-only rehearsal of the distributed run, before any GPU is rented (integration_tests.md, R). From the laptop:
# one real sandbox box (scripts/ec2_sandboxes.sh up 1), plus a small instance standing in for the H200 that runs
# the real links.sh (tunnels, trial sync, dispatcher), the tracker, and scripts/stub_model.py in place of Miles.
# Then 4 real Harbor trials go through dispatcher -> tunnel -> box -> terminus-2 -> reverse tunnel -> stub -> verifier.
#
#   bash scripts/rehearse.sh            # ~10-15 min, ~$0.40; terminates both instances at the end (KEEP=1 keeps them)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
export STATE=$HERE/state/rehearsal REGION=${REGION:-us-east-2} TTL_MIN=${TTL_MIN:-120}
KEY=${KEY:-$HOME/.ssh/aees-ec2.pem}
source "$HERE/scripts/ec2_lib.sh"
mkdir -p "$STATE"
RUN=rehearsal
T0=$(date +%s); el() { echo "[$(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s] $*"; }
SSH_S() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o BatchMode=yes -x "ubuntu@$SIP" "$@"; }
cleanup() {
  [ -n "${KEEP:-}" ] && { echo "KEEP=1: instances left running (they terminate after $TTL_MIN min)"; return; }
  bash "$HERE/scripts/ec2_sandboxes.sh" down || true
  [ -n "${SID:-}" ] && aws ec2 terminate-instances --region "$REGION" --instance-ids "$SID" --output text >/dev/null || true
}
trap cleanup EXIT

el "1/6 sandbox box + H200 stand-in (c7a.large), in parallel"
bash "$HERE/scripts/ec2_sandboxes.sh" up 1 > "$STATE/up.log" 2>&1 & up=$!
SG=$(ec2_sg "$REGION" aees-grpo "aees GRPO sandboxes: SSH from the laptop and the H200 only")
SID=$(aws ec2 run-instances --region "$REGION" --image-id "$(ec2_ami "$REGION")" --instance-type c7a.large \
  --key-name "${KEY_NAME:-aees-ec2}" --security-group-ids "$SG" --instance-initiated-shutdown-behavior terminate \
  --user-data "$(printf '#!/bin/bash\nshutdown -h +%s\n' "$TTL_MIN")" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aees-h200-standin},{Key=Project,Value=aees}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REGION" --instance-ids "$SID"
SIP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$SID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ec2_wait_ssh "$KEY" "$SIP"
SSH_S 'mkdir -p geodesic/aees/post_training geodesic/aees/harbor/state && command -v tmux rsync curl python3 >/dev/null && echo stand-in ok'
tar -C "$HERE" --exclude ./.venv --exclude ./data --exclude ./state --exclude __pycache__ --exclude .pytest_cache \
  --exclude ./proto/state -czf - . | SSH_S 'tar -xzf - -C geodesic/aees/post_training'
tar -C "$HERE/.." -czf - harbor/state/tb21_selected_tasks.txt harbor/state/tb21_val8_tasks.txt harbor/state/tb21_test20_tasks.txt \
  | SSH_S 'tar -xzf - -C geodesic/aees'
wait "$up" || { el "box prepare failed:"; tail -20 "$STATE/up.log"; tail -30 "$STATE"/box_1.log; exit 1; }
el "    $(tail -1 "$STATE/up.log")"

el "2/6 allow the stand-in, agent server in t3 mode"
H200_KEY=$KEY bash "$HERE/scripts/ec2_sandboxes.sh" allow "$SIP"
bash "$HERE/scripts/ec2_sandboxes.sh" serve "$RUN" t3

el "3/6 stand-in: stub model on :30000, links, tracker"
SSH_S "cd geodesic/aees/post_training && tmux new -d -s stub 'python3 scripts/stub_model.py --port 30000' && \
  sudo mkdir -p /data && sudo chown ubuntu /data && RUN=$RUN bash scripts/links.sh start && \
  tmux new -d -s tracker 'RUN=$RUN NUM_ROLLOUT=1 python3 monitor/server.py'"
sleep 5

el "4/6 reverse tunnels"
bash "$HERE/scripts/ec2_sandboxes.sh" probe

el "5/6 4 real trials through the dispatcher"
SSH_S "python3 - <<'PY'
import json, threading, urllib.request
def run(task, out):
    body = json.dumps({'instance_id': task, 'agent_name': 'terminus-2', 'base_url': 'http://127.0.0.1:30000/v1',
                       'model': 'openai/stub', 'sampling_params': {}}).encode()
    req = urllib.request.Request('http://127.0.0.1:11000/run', data=body, headers={'Content-Type': 'application/json'})
    out.append((task, json.loads(urllib.request.urlopen(req, timeout=900).read())))
out = []
ts = [threading.Thread(target=run, args=(t, out)) for t in ['configure-git-webserver', 'sqlite-with-gcov'] * 2]
[t.start() for t in ts]; [t.join() for t in ts]
for task, r in out: print(task, r.get('exit_status'), r.get('reward'), (r.get('agent_metrics') or {}).get('exception_type'))
PY"

el "6/6 trial sync + tracker (waits one sync interval)"
sleep 70
SSH_S "ls /data/trials/$RUN/*/result.json 2>/dev/null | wc -l | sed 's/^/synced result.json: /'; \
  wc -l < /data/trials/$RUN/dispatch.jsonl | sed 's/^/dispatch rows: /'; \
  curl -s 127.0.0.1:8770/status.json | python3 -c 'import json,sys; s=json.load(sys.stdin); \
  print(\"tracker:\", len(s[\"trials\"]), \"trials,\", len(s[\"dispatch\"]), \"dispatch rows, sandboxes:\", (s[\"sandboxes\"] or {}).get(\"boxes_up\"))'"
el "done"
