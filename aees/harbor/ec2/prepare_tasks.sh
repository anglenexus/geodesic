#!/usr/bin/env bash
# Prepare an EC2 instance (after setup_ec2.sh) for the TB 2.1 collection:
#   1. check Docker can create enough networks for the planned concurrency
#   2. download TB 2.1 task definitions and pre-pull the selected tasks' images
#   3. run the oracle on the 20 selected tasks + the 4 deferred (Rosetta-blocked) tasks
#   4. check the OpenAI key with a 4-token request
# Safe to re-run. Takes ~15-40 min, mostly image pulls.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

HERE=$(cd "$(dirname "$0")/.." && pwd)        # aees/harbor
CONCURRENCY=${CONCURRENCY:-100}
TB21_DIR=${TB21_DIR:-$HOME/tb21}
JOBS=${JOBS:-$HOME/jobs}
DEFERRED="qemu-startup qemu-alpine-ssh tune-mjcf bn-fit-modify"

echo "== 1/4 network capacity: creating $((CONCURRENCY + 20)) test networks"
created=0
for i in $(seq 1 $((CONCURRENCY + 20))); do
  if docker network create "capacity-test-$i" >/dev/null 2>&1; then created=$((created + 1)); else break; fi
done
docker network ls -q --filter name=capacity-test- | xargs -r docker network rm >/dev/null
if [ "$created" -lt $((CONCURRENCY + 20)) ]; then
  echo "FAIL: only $created networks could be created; check default-address-pools in /etc/docker/daemon.json"
  exit 1
fi
echo "ok: $created networks"

echo "== 2/4 task definitions + image pre-pull"
[ -d "$TB21_DIR" ] || harbor datasets download terminal-bench/terminal-bench-2-1 -o "$TB21_DIR"
python3 "$HERE/scripts/task_images.py" "$TB21_DIR" "$HERE/state/tb21_selected_tasks.txt" > /tmp/tb21_images.txt
printf '%s\n' $DEFERRED > /tmp/tb21_deferred.txt
python3 "$HERE/scripts/task_images.py" "$TB21_DIR" /tmp/tb21_deferred.txt >> /tmp/tb21_images.txt
xargs -P 8 -n 1 docker pull -q < /tmp/tb21_images.txt
echo "pulled $(wc -l < /tmp/tb21_images.txt) images; disk free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"

echo "== 3/4 oracle check (no API cost)"
args=()
for t in $(cat "$HERE/state/tb21_selected_tasks.txt") $DEFERRED; do args+=(-i "terminal-bench/$t"); done
harbor run --dataset terminal-bench/terminal-bench-2-1 --agent oracle --env docker \
  -o "$JOBS" --job-name "ec2_oracle_$(date +%Y%m%d_%H%M)" -n 24 -r 0 -y "${args[@]}" | tail -15

echo "== 4/4 OpenAI key check"
KEY=$(grep '^OPENAI_API_KEY=' "$HOME/.config/harbor/.env" | cut -d= -f2-)
curl -s https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-luna","messages":[{"role":"user","content":"OK"}],"max_completion_tokens":4}' \
  | jq -r 'if .choices then "API ok" else "API ERROR: \(.error.code) \(.error.message)" end'

cat <<'NEXT'

Review the oracle results above: all 20 selected tasks should score 1.0. The 4 deferred tasks
(qemu-startup, qemu-alpine-ssh, tune-mjcf, bn-fit-modify) failed only under Rosetta on the Mac;
record their x86 result in state/tb21_task_notes.md.
Then follow "Run the collection" in ec2/EC2.md.
NEXT
