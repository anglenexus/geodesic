#!/usr/bin/env bash
# serve_kimi_k3.sh - run Kimi K3 with SGLang on one 8x B300 node, in a tmux session, for the Phase 3/4 tests.
# Runs inside the Modal container started by modal_kimi_k3.py (SGLang image, weights in a Volume at
# /models/Kimi-K3); same interface as serve_qwen3_27B.sh, so run_bench.py drives both the same way.
#
# Usage:  ./serve_kimi_k3.sh start | stop | restart | status | logs | tunnel
#
# Profile: SGLang's verified B300 1x8 "Balanced" recipe: TP8 with decode context parallel 8 (DCP=8 shards the
# MLA latent cache by tokens across the 8 GPUs instead of keeping a full copy on each). DCP=1 gives the
# "Low-Latency" recipe. Weights are the native MXFP4 checkpoint (routed experts 4-bit, the rest BF16);
# SGLang reads the format from the checkpoint and picks FlashInfer's MXFP4 MoE and trtllm_mla kernels.
#
# Options (env vars):
#   MODEL_DIR=/models/Kimi-K3  SERVED_NAME=kimi-k3  HOST=127.0.0.1  PORT=30000
#   TP=8  DCP=8  MEM_FRACTION=0.85  CONTEXT_LEN=65536  MAX_RUNNING=128
#   EXTRA_ARGS="..."   extra sglang flags (run_bench.py passes its settings here), e.g.
#                      "--mamba-ssm-dtype bfloat16" or "--kv-cache-dtype fp8_e4m3"
#   SERVER_ENV="VAR=..."  extra environment for the server process
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/kimi-serve}"
MODEL_DIR="${MODEL_DIR:-/models/Kimi-K3}"
SERVED_NAME="${SERVED_NAME:-kimi-k3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DCP="${DCP:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LEN="${CONTEXT_LEN:-65536}"
MAX_RUNNING="${MAX_RUNNING:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Weights on a network Volume: by default each of the 8 ranks reads all 1.56 TB (12.5 TB at the Volume's ~2 GB/s,
# ~60 min). With prefetch each rank reads 1/8 of the shards into the shared page cache and the others hit it:
# 1.56 TB once, ~13 min. One thread per rank = 8 parallel streams, the fastest measured (16 streams: 0.9 GB/s).
LOAD_FLAGS="${LOAD_FLAGS---weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 1}"
SERVER_ENV="${SERVER_ENV:-}"
SESSION="${SESSION:-sglang}"
LOG_DIR="$WORK_DIR/logs"
STARTUP_MIN="${STARTUP_MIN:-40}"  # 1.56 TB from the Volume at ~2 GB/s is ~13 min, then kernels + CUDA graphs
URL="http://127.0.0.1:$PORT"

running() { tmux has-session -t "$SESSION" 2>/dev/null; }
ready()   { curl -sf "$URL/v1/models" >/dev/null 2>&1; }

summary() {
  local log="$LOG_DIR/latest.log"
  echo "--- startup facts (record these in the run log) ---"
  grep -m3 -iE "moe.?runner|mxfp4|trtllm|quantization" "$log" || true
  grep -m2 -iE "attention.?backend|mla" "$log" || true
  grep -m1 -iE "load weight end" "$log" || true
  grep -m1 -E "max_total_num_tokens" "$log" || true
  grep -m3 -iE "mamba|max_running_requests" "$log" || true
  grep -m1 -iE "cuda graph" "$log" || true
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  echo "--- test request (thinking off) ---"
  curl -s "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print('reply:', r['choices'][0]['message']['content'])" || true
  echo "Endpoint: http://$HOST:$PORT/v1   model name: $SERVED_NAME   TP=$TP DCP=$DCP"
  tunnel
}

tunnel() {
  echo "--- from your laptop: use the ssh command printed at the top of the Modal run log ---"
  echo "  API: http://127.0.0.1:$PORT/v1   (python smoke_test.py --model $SERVED_NAME --no-think)"
}

start() {
  if running; then echo "A server is already running (tmux session '$SESSION'). Use: $0 restart"; return; fi
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found at $MODEL_DIR; run: modal run modal_kimi_k3.py::download"; exit 1; }
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if (( used > 4096 )); then
    echo "A GPU already has ${used} MiB in use; Kimi K3 needs all 8. Stop the other process first:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  local log; log="$LOG_DIR/server_kimi_k3_dcp${DCP}_$(date +%Y%m%d_%H%M%S).log"
  local par="--tp-size $TP"
  (( DCP > 1 )) && par+=" --dcp-size $DCP"
  # The exact command is written to a file so every run is reproducible from the log dir.
  cat > "$WORK_DIR/.run_server_kimi.sh" <<EOF
#!/usr/bin/env bash
${SERVER_ENV:+export $SERVER_ENV}
sglang serve --model-path "$MODEL_DIR" --trust-remote-code \\
  --served-model-name $SERVED_NAME \\
  --host $HOST --port $PORT \\
  $par \\
  --mem-fraction-static $MEM_FRACTION $LOAD_FLAGS \\
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \\
  --context-length $CONTEXT_LEN \\
  --max-running-requests $MAX_RUNNING \\
  --enable-metrics \\
  --enable-cache-report \\
  --log-level info $EXTRA_ARGS 2>&1 | tee "$log"
EOF
  chmod 700 "$WORK_DIR/.run_server_kimi.sh"
  cp "$WORK_DIR/.run_server_kimi.sh" "${log%.log}.cmd"
  ln -sf "$log" "$LOG_DIR/latest.log"
  tmux new-session -d -s "$SESSION" "$WORK_DIR/.run_server_kimi.sh"

  echo "Starting (TP=$TP DCP=$DCP); log: $log"
  for i in $(seq 1 $((STARTUP_MIN * 12))); do
    if ready; then echo "Ready after ~$((i * 5)) s"; summary; return; fi
    if ! running; then
      echo "Server exited during startup. Last log lines:"; tail -n 60 "$log"
      exit 1
    fi
    (( i % 12 == 0 )) && echo "  ... $((i / 12)) min: $(grep -c -iE 'loading safetensors|load weight' "$log" || true) load lines so far"
    sleep 5
  done
  echo "Not ready after $STARTUP_MIN minutes; check: $0 logs"
  exit 1
}

stop() {
  if running; then
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 1 60); do running || break; sleep 1; done
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
  pkill -f "sglang serve" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "sglang" >/dev/null || break; sleep 2; done
  echo "Stopped."
}

status() {
  if running; then echo "tmux session '$SESSION': running"; else echo "tmux session '$SESSION': not running"; fi
  if ready; then echo "API: ready at $URL/v1"; else echo "API: not responding"; fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 3; start ;;
  status)  status ;;
  logs)    tail -n 100 -f "$LOG_DIR/latest.log" ;;
  tunnel)  tunnel ;;
  *)       echo "Usage: $0 {start|stop|restart|status|logs|tunnel}"; exit 1 ;;
esac
