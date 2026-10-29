#!/usr/bin/env bash
# On the H200: integration tests T0, T2, T3 (integration_tests.md). Each check appends one row
# {"t", "test", "check", "ok", "value"} to $DATA/monitor/$RUN/preflight.jsonl (the tracker's Preflight card).
#
#   RUN=grpo_10h bash scripts/preflight.sh t0      # infrastructure, ~2 min; after links.sh start
#   RUN=grpo_10h bash scripts/preflight.sh t2      # SGLang at 96 trials, 6-8 min; GPU must be free
#   RUN=grpo_10h bash scripts/preflight.sh t3      # mini GRPO, <= 10 min; boxes serving in t3 mode, RUN_T3 links
# Exit code 0 only if every check of that test passed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
source "$HERE/configs/pins.sh"
DATA=${DATA:-/data}
RUN=${RUN:?set RUN, e.g. RUN=grpo_10h}
MODEL_DIR=$DATA/models/Qwen3.6-27B
OUT=$DATA/monitor/$RUN
mkdir -p "$OUT"
FAILED=0
T2_AGENTS=${T2_AGENTS:-96}
T2_MIN=${T2_MIN:-3}

record() {            # record <test> <check> <ok 0/1> <value>
  python3 -c 'import json,sys,time; print(json.dumps({"t": time.time(), "test": sys.argv[1], "check": sys.argv[2],
    "ok": sys.argv[3] == "1", "value": sys.argv[4]}))' "$1" "$2" "$3" "$4" >> "$OUT/preflight.jsonl"
  [ "$3" = 1 ] && echo "  ok    $2: $4" || { echo "  FAIL  $2: $4"; FAILED=1; }
}
check() {             # check <test> <check> <command...>: ok if the command succeeds; value = its last output line
  local out; out=$("${@:3}" 2>&1); local rc=$?
  record "$1" "$2" "$([ $rc = 0 ] && echo 1 || echo 0)" "$(echo "$out" | tail -1 | cut -c1-200)"
}
miles() { docker run --rm --gpus all --ipc=host --network host -v "$DATA/models:/root/models" -v "$DATA/cache:/root/.cache" \
  -v "$DATA/proto:/root/proto_out" -v "$HERE:/root/post_training" -w /root/post_training -e PYTHONPATH=/root/post_training \
  "$@"; }
sglang_server_flags() {   # SGLang flags as Miles would pass them (configs/grpo.sh), without the --sglang- prefix
  ( SAVE_DIR=/tmp/preflight MAX_LEN=${MAX_LEN:-49152}; source "$HERE/configs/grpo.sh"; printf '%s\n' "${SGLANG_SERVER[@]}" )
}

t0() {
  echo "== T0 infrastructure"
  local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  record T0 "driver >= 580" "$([ "${drv%%.*}" -ge 580 ] 2>/dev/null && echo 1 || echo 0)" "$drv"
  check T0 "GPU in the Miles container" miles "$MILES_IMAGE" nvidia-smi -L
  check T0 "weights" bash -c "compgen -G '$MODEL_DIR/*.safetensors' >/dev/null && du -sh $MODEL_DIR | cut -f1"
  if [ -d "${MODEL_DIR}_torch_dist" ]; then record T0 "torch_dist checkpoint" 1 "${MODEL_DIR}_torch_dist"
  else record T0 "torch_dist checkpoint" 1 "pending (setup.sh step 9 still running?)"; fi
  local free; free=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
  record T0 "disk free >= 400 GB" "$([ "$free" -ge 400 ] && echo 1 || echo 0)" "${free} GB"
  local help missing=""
  help=$(docker run --rm "$MILES_IMAGE" python3 -m sglang.launch_server --help 2>&1)
  for f in $(sglang_server_flags | grep '^--') --enable-lora --max-lora-rank --lora-target-modules --lora-paths; do
    grep -q -- "$f" <<<"$help" || missing+="$f "
  done
  record T0 "SGLang knows our flags" "$([ -z "$missing" ] && echo 1 || echo 0)" "${missing:-all present}"
  local i=0 ip
  for ip in $(cat "${BOXES_FILE:-$HOME/aees_boxes}"); do
    i=$((i + 1))
    check T0 "box $i agent server via tunnel ($ip)" curl -sf --max-time 5 "127.0.0.1:1100$i/health"
  done
  check T0 "dispatcher: all boxes up" bash -c "curl -sf --max-time 20 127.0.0.1:11000/health | python3 -c \
    'import json,sys; h=json.load(sys.stdin); up=h[\"boxes_up\"]; print(sum(up.values()), \"of\", len(up), \"boxes,\", \
    sum(b[\"free\"] for b in h[\"boxes\"].values()), \"slots\"); sys.exit(not all(up.values()))'"
  check T0 "tracker answers" curl -sf --max-time 5 -o /dev/null -w '%{http_code}' 127.0.0.1:8770/status.json
  # The reverse tunnels are checked from the laptop (scripts/ec2_sandboxes.sh probe) against this stub:
  if ! curl -s --max-time 2 127.0.0.1:30000 >/dev/null 2>&1; then
    (cd /tmp && timeout 300 python3 -c 'import http.server as h
class H(h.BaseHTTPRequestHandler):
    def do_GET(s): s.send_response(200); s.end_headers(); s.wfile.write(b"aees-stub")
    def log_message(s, *a): pass
h.HTTPServer(("127.0.0.1", 30000), H).serve_forever()' &)
    echo "  stub on 127.0.0.1:30000 for 5 min: run scripts/ec2_sandboxes.sh probe on the laptop"
  fi
}

t2() {
  echo "== T2 SGLang at $T2_AGENTS trials"
  local dir=$DATA/proto/t2_$RUN log
  mkdir -p "$dir" "$DATA/cache"
  log=$dir/sglang.log
  [ -f "$dir/adapter/adapter_config.json" ] || miles "$MILES_IMAGE" python3 -m proto.adapter /root/models/Qwen3.6-27B \
    "/root/proto_out/t2_$RUN/adapter" > "$dir/adapter.log" 2>&1
  local targets; targets=$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["target_modules"]))' \
    "$dir/adapter/adapter_config.json")
  docker rm -f t2 >/dev/null 2>&1
  pkill -f "aees-stub" 2>/dev/null; sleep 1                # T0's stub holds port 30000
  local t_start; t_start=$(date +%s)
  miles -d --name t2 "$MILES_IMAGE" bash -c "python3 -m sglang.launch_server --model-path /root/models/Qwen3.6-27B \
    --served-model-name qwen3.6-27b --host 127.0.0.1 --port 30000 $(sglang_server_flags | tr '\n' ' ') \
    --enable-lora --max-lora-rank 64 --lora-target-modules $targets --lora-paths t2=/root/proto_out/t2_$RUN/adapter \
    --enable-cache-report --enable-metrics > /root/proto_out/t2_$RUN/sglang.log 2>&1" >/dev/null
  local up=0
  for _ in $(seq 120); do
    docker ps -q -f name=t2 | grep -q . || break
    curl -sf 127.0.0.1:30000/health_generate >/dev/null 2>&1 && { up=1; break; }
    sleep 3
  done
  record T2 "SGLang starts with our flags + adapter" "$up" "$(( $(date +%s) - t_start )) s"
  if [ "$up" = 1 ]; then
    local cap; cap=$(grep -oiE "max_running_requests[^,]*|capped to [0-9]+[^\"]*" "$log" | tail -1)
    local maxrun; maxrun=$(grep -oE "max_running_requests[= :]+[0-9]+" "$log" | grep -oE "[0-9]+$" | tail -1)
    record T2 "running-request cap >= 64" "$([ "${maxrun:-0}" -ge 64 ] && echo 1 || echo 0)" "${cap:-not in log} | $(grep -oE "max_total_num_tokens=[0-9]+" "$log" | tail -1)"
    (while docker ps -q -f name=t2 | grep -q .; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits; sleep 5; done > "$dir/gpu.csv") &
    miles "$MILES_IMAGE" python3 -m proto.agent_load --profile proto/load_profile.json --agents "$T2_AGENTS" \
      --minutes "$T2_MIN" --stagger 0.5 --lora t2 --out "/root/proto_out/t2_$RUN/agents.jsonl" > "$dir/load.log" 2>&1
    local stats; stats=$(cd "$HERE" && python3 -c '
import json, sys
from proto.report import agent_stats, jsonl
from pathlib import Path
rows = jsonl(Path(sys.argv[1]))
t0 = min(r["t_send"] for r in rows) + 30                    # first 30 s: agents ramp up, caches fill
s = agent_stats(rows, t0, float("inf"), True, 0)
sel = [r for r in rows if r.get("t_first") and r["t_send"] >= t0 and not r.get("error")]
hit = sum(r["cached"] for r in sel) / max(1, sum(r["prompt"] for r in sel))
ttft = sorted(r["t_first"] - r["t_send"] for r in sel)
gpu = [list(map(float, l.split(","))) for l in open(sys.argv[2]) if l.strip()]
used, total = max(g[0] for g in gpu) / 1024, gpu[0][1] / 1024
print(json.dumps({"hit": round(hit, 3), "ttft_p90": round(ttft[int(.9 * len(ttft))], 2) if ttft else None,
                  "calls": len(sel), "errors": s.get("errors", 0), "headroom_gb": round(total - used, 1),
                  "per_tok_ms_p50": s.get("per_tok_ms_p50"), "out_tok_s": s.get("out_tok_s")}))' \
      "$dir/agents.jsonl" "$dir/gpu.csv" 2>&1)
    local v; v() { python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2]))' "$stats" "$1" 2>/dev/null; }
    record T2 "prefix-cache hit >= 0.60" "$(python3 -c "import sys; sys.exit(0 if float('$(v hit)') >= .6 else 1)" 2>/dev/null && echo 1 || echo 0)" "$stats"
    record T2 "TTFT p90 <= 5 s" "$(python3 -c "import sys; sys.exit(0 if float('$(v ttft_p90)') <= 5 else 1)" 2>/dev/null && echo 1 || echo 0)" "$(v ttft_p90) s, $(v calls) calls, $(v errors) errors"
    record T2 "headroom >= 8 GB, no OOM" "$(python3 -c "import sys; sys.exit(0 if float('$(v headroom_gb)') >= 8 else 1)" 2>/dev/null && ! grep -qi "out of memory" "$log" && echo 1 || echo 0)" "$(v headroom_gb) GB"
  else
    tail -20 "$log"
  fi
  docker rm -f t2 >/dev/null 2>&1
}

t3() {
  echo "== T3 mini GRPO (2 tasks x 4, 2 steps, 2-turn trials)"
  local save=$DATA/ckpt/$RUN
  [ ! -e "$save" ] || { echo "$save exists; use a new RUN for T3"; exit 1; }
  python3 -c 'import json,sys; sys.path.insert(0, sys.argv[1]); from grpo.split_tasks import row
open(sys.argv[2], "w").write("".join(json.dumps(row(t)) + "\n" for t in ["configure-git-webserver", "sqlite-with-gcov"]))' \
    "$HERE" "$HERE/data/grpo_t3.jsonl"
  local t_start; t_start=$(date +%s)
  tmux kill-session -t t3 2>/dev/null
  tmux new -d -s t3 "SAVE_DIR=/root/ckpt/$RUN NUM_ROLLOUT=2 ROLLOUT_BATCH=2 N_SAMPLES=4 GLOBAL_BATCH=8 SAVE_INTERVAL=1 \
    PROMPT_DATA=/root/post_training/data/grpo_t3.jsonl SGLANG_EXTRA='${SGLANG_EXTRA:-}' DATA=$DATA \
    bash $HERE/scripts/run.sh grpo > $OUT/t3_run.log 2>&1"
  local done=0
  for _ in $(seq $(( ${T3_MAX_MIN:-15} * 6 ))); do
    [ -d "$save/iter_0000001" ] && { done=1; break; }
    tmux has-session -t t3 2>/dev/null || break
    sleep 10
  done
  sleep 5
  record T3 "2 steps trained and saved" "$done" "$(( ($(date +%s) - t_start) / 60 )) min; $(ls -d "$save"/iter_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
  local rows; rows=$(wc -l < "$DATA/trials/$RUN/dispatch.jsonl" 2>/dev/null || echo 0)
  record T3 "16 trials through the dispatcher" "$([ "$rows" -ge 16 ] && echo 1 || echo 0)" "$rows rows"
  local infra; infra=$(cat "$DATA/trials/$RUN/dispatch.jsonl" 2>/dev/null | grep -c '"exit_status": "Error')
  record T3 "no infra errors" "$([ "$infra" = 0 ] && echo 1 || echo 0)" "$infra"
  local synced; synced=$(ls "$DATA"/trials/"$RUN"/*/result.json 2>/dev/null | wc -l)
  record T3 "trials synced to the H200" "$([ "$synced" -ge 16 ] && echo 1 || echo 0)" "$synced result.json"
  local steps; steps=$(cat "$save"/events/*.jsonl 2>/dev/null | grep -o '"train/[a-z_]*loss[a-z_]*": *[-0-9.e]*' | wc -l)
  record T3 "training metrics logged" "$([ "$steps" -ge 2 ] && echo 1 || echo 0)" "$steps loss values"
  check T3 "tracker shows the steps" bash -c "curl -sf 127.0.0.1:8770/status.json | python3 -c \
    'import json,sys; s=json.load(sys.stdin); print(len(s[\"steps\"]), \"steps,\", len(s[\"trials\"]), \"trials\"); sys.exit(len(s[\"steps\"]) < 2)'"
  [ "$done" = 1 ] || tail -30 "$OUT/t3_run.log"
  tmux kill-session -t t3 2>/dev/null
  docker ps -q --filter "ancestor=$MILES_IMAGE" | xargs -r docker rm -f >/dev/null   # a hung trainer, if any
}

case "${1:-}" in
t0|t2|t3) "$1"; exit "$FAILED" ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
