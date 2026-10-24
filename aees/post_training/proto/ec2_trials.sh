#!/usr/bin/env bash
# EC2 side of the overlap prototype (PLAN.md §5b): N real terminus-2 trials in flight against the H200's
# SGLang for the whole bench run, through an SSH tunnel and llm_log_proxy.py (per-call log: calls.jsonl).
# Started by proto/kickoff.sh in tmux. Waits for the server, then keeps N trials running until MAX_MIN
# minutes have passed or the server has been down DOWN_S seconds in a row (bench.sh finished).
#
#   H200=<ip> H200_USER=<user> RUN=<run> bash ~/proto/ec2_trials.sh
set -uo pipefail
N=${N:-8}
MAX_MIN=${MAX_MIN:-120}
DOWN_S=${DOWN_S:-480}                    # longer than a server restart between phases (~3-5 min)
POLL_S=${POLL_S:-15}
STAGGER_S=${STAGGER_S:-20}
OUT=$HOME/proto/out
TASKS=(); while read -r t; do [ -n "$t" ] && TASKS+=("$t"); done < "$HOME/proto/ec2_tasks.txt"
mkdir -p "$OUT" "$HOME/proto/jobs"
export PATH="$HOME/.local/bin:$PATH" HOSTED_VLLM_API_KEY=none
deadline=$(( $(date +%s) + MAX_MIN * 60 ))

# Tunnel: localhost:30001 -> the H200's 127.0.0.1:30000. The key is only allowed this one forward there.
( while :; do
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o StrictHostKeyChecking=accept-new \
      -i "$HOME/.ssh/proto_tunnel" -L 30001:127.0.0.1:30000 "$H200_USER@$H200"
    sleep 5
  done ) >> "$OUT/tunnel.log" 2>&1 &
python3 "$HOME/proto/llm_log_proxy.py" --port 8001 --upstream http://127.0.0.1:30001 --out "$OUT/calls.jsonl" \
  >> "$OUT/proxy.log" 2>&1 &
cleanup() {           # only this run's harbor jobs; the box is dedicated, so every container goes
  rm -f "$OUT/running"
  pkill -INT -f "harbor run.*--job-name proto_${RUN}_"
  sleep "$POLL_S"
  docker ps -q | xargs docker rm -f >/dev/null 2>&1
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT
up() { curl -sf -m 5 http://127.0.0.1:30001/health >/dev/null; }

until up; do [ "$(date +%s)" -lt "$deadline" ] || exit 1; sleep "$POLL_S"; done
echo "$(date +%T) server up; running $N trials at a time"
worker() {            # one slot: trials back to back, tasks round-robin
  local w=$1 k=0 t left
  while [ -f "$OUT/running" ]; do
    t=${TASKS[$(( (w + k * N) % ${#TASKS[@]} ))]}; k=$((k + 1))
    up || { sleep "$POLL_S"; continue; }
    left=$(( deadline - $(date +%s) )); [ "$left" -gt 10 ] || break
    timeout --signal=INT "$left" harbor run --dataset terminal-bench/terminal-bench-2-1 --agent terminus-2 \
      -m hosted_vllm/qwen3.6-27b --ak api_base='"http://127.0.0.1:8001/v1"' \
      --ak 'model_info={"max_input_tokens": 49152, "max_output_tokens": 8192, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}' \
      --ak 'llm_call_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}' \
      --env docker -o "$HOME/proto/jobs" --job-name "proto_${RUN}_s${w}_$k" -n 1 -r 0 -y -i "terminal-bench/$t" \
      >> "$OUT/harbor_s$w.log" 2>&1
    echo "$(date +%T) slot $w: $t done ($?)"
  done
}
touch "$OUT/running"
for w in $(seq 0 $((N - 1))); do worker "$w" & sleep "$STAGGER_S"; done      # staggered starts
down_since=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if up; then down_since=0
  else
    [ "$down_since" != 0 ] || down_since=$(date +%s)
    [ $(( $(date +%s) - down_since )) -lt "$DOWN_S" ] || { echo "$(date +%T) server down ${DOWN_S} s: stopping"; break; }
  fi
  sleep "$POLL_S"
done
