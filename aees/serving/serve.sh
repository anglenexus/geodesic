#!/usr/bin/env bash
# serve.sh - run the Qwen3-4B SGLang server with the plan's baseline flags in a tmux session.
#
# Usage:  ./serve.sh start | stop | restart | status | logs
#
# Options (env vars):
#   HOST=127.0.0.1  PORT=30000      bind address (use an SSH tunnel, or 0.0.0.0 + API_KEY + firewall rule)
#   API_KEY=...                     require a bearer token on the API
#   MEM_FRACTION=0.85  CONTEXT_LEN=32768
#   TOOL_PARSER=qwen25              qwen3_coder for Qwen3.5 models
#   REASONING_PARSER=               qwen3 for the hybrid Qwen/Qwen3-4B
#   EXTRA_ARGS="--kv-cache-dtype fp8_e5m2"   any extra sglang flags (experiment arms)
#   MODEL_ID / MODEL_DIR / WORK_DIR  must match what setup_host.sh used
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/${MODEL_ID##*/}}"
SERVED_NAME="${SERVED_NAME:-qwen3-4b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
TOOL_PARSER="${TOOL_PARSER:-qwen25}"
REASONING_PARSER="${REASONING_PARSER:-}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SESSION="${SESSION:-sglang}"
CUDA_HOME_DIR="/usr/local/cuda-13.0"
LOG_DIR="$WORK_DIR/logs"
URL="http://127.0.0.1:$PORT"
AUTH=()
if [[ -n "$API_KEY" ]]; then AUTH=(-H "Authorization: Bearer $API_KEY"); fi

running() { tmux has-session -t "$SESSION" 2>/dev/null; }
ready()   { curl -sf "${AUTH[@]}" "$URL/v1/models" >/dev/null 2>&1; }

summary() {
  local log="$LOG_DIR/latest.log"
  echo "--- startup facts (record these in the run log) ---"
  grep -m1 -iE "attention.?backend" "$log" || true
  grep -m1 -E "max_total_num_tokens" "$log" || true
  grep -m1 -iE "cuda graph" "$log" || true
  echo "--- test request ---"
  curl -s "${AUTH[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print('reply:', r['choices'][0]['message']['content'])"
  echo "Endpoint: http://$HOST:$PORT/v1   model name: $SERVED_NAME"
  if [[ "$HOST" == "127.0.0.1" ]]; then
    echo "From a laptop: ssh -N -L $PORT:localhost:$PORT $(whoami)@<this-host-ip>"
  fi
}

start() {
  if running; then echo "Already running (tmux session '$SESSION'). Use: $0 restart"; return; fi
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found at $MODEL_DIR; run setup_host.sh first."; exit 1; }
  [[ -d "$WORK_DIR/.venv" ]] || { echo "No venv at $WORK_DIR/.venv; run setup_host.sh first."; exit 1; }
  mkdir -p "$LOG_DIR"
  local log; log="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
  local opt=""
  [[ -z "$API_KEY" ]] || opt+=" --api-key $API_KEY"
  [[ -z "$REASONING_PARSER" ]] || opt+=" --reasoning-parser $REASONING_PARSER"

  # The exact command is written to a file so every run is reproducible from the log dir.
  cat > "$WORK_DIR/.run_server.sh" <<EOF
#!/usr/bin/env bash
source "$WORK_DIR/.venv/bin/activate"
export CUDA_HOME=$CUDA_HOME_DIR PATH=$CUDA_HOME_DIR/bin:\$PATH LD_LIBRARY_PATH=$CUDA_HOME_DIR/lib64:\${LD_LIBRARY_PATH:-}
sglang serve "$MODEL_DIR" \\
  --served-model-name $SERVED_NAME \\
  --host $HOST --port $PORT \\
  --attention-backend flashinfer \\
  --sampling-backend flashinfer \\
  --tool-call-parser $TOOL_PARSER \\
  --context-length $CONTEXT_LEN \\
  --mem-fraction-static $MEM_FRACTION \\
  --chunked-prefill-size 8192 \\
  --enable-metrics \\
  --enable-cache-report \\
  --log-level info$opt $EXTRA_ARGS 2>&1 | tee "$log"
EOF
  chmod 700 "$WORK_DIR/.run_server.sh"
  grep -v -- "--api-key" "$WORK_DIR/.run_server.sh" > "${log%.log}.cmd" || true
  ln -sf "$log" "$LOG_DIR/latest.log"
  tmux new-session -d -s "$SESSION" "$WORK_DIR/.run_server.sh"

  echo "Starting; log: $log"
  for i in $(seq 1 120); do
    if ready; then echo "Ready after ~$((i * 5)) s"; summary; return; fi
    if ! running; then echo "Server exited during startup. Last log lines:"; tail -n 40 "$log"; exit 1; fi
    sleep 5
  done
  echo "Not ready after 10 minutes; check: $0 logs"
  exit 1
}

stop() {
  if running; then
    tmux send-keys -t "$SESSION" C-c
    for _ in $(seq 1 30); do running || break; sleep 1; done
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
  pkill -f "sglang serve" 2>/dev/null || true
  echo "Stopped."
}

status() {
  if running; then echo "tmux session '$SESSION': running"; else echo "tmux session '$SESSION': not running"; fi
  if ready; then echo "API: ready at $URL/v1"; else echo "API: not responding"; fi
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 3; start ;;
  status)  status ;;
  logs)    tail -n 100 -f "$LOG_DIR/latest.log" ;;
  *)       echo "Usage: $0 {start|stop|restart|status|logs}"; exit 1 ;;
esac
