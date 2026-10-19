#!/usr/bin/env bash
# serve_qwen3_27B.sh - run Qwen3.5-27B (4-bit AWQ) with SGLang on one A100-40GB, in a tmux session,
# as the model behind Harbor runs. Same port and tunnel as serve.sh, so Harbor only changes the
# model name (-m hosted_vllm/qwen3.5-27b). Only one of the two servers fits on the GPU at a time.
#
# Usage:  ./serve_qwen3_27B.sh start | stop | restart | status | logs | tunnel
#
# Memory on 40 GB: ~20 GB of weights; the rest holds the KV cache of the 16 full-attention layers
# (64 KiB/token) and a fixed ~150 MiB linear-attention state per running request (48 layers).
# MAX_RUNNING caps the latter; it should be at least Harbor's -n.
#
# Options (env vars):
#   HOST=127.0.0.1  PORT=30000      bind address (use an SSH tunnel, or 0.0.0.0 + API_KEY + firewall rule)
#   API_KEY=...                     require a bearer token on the API
#   MEM_FRACTION=0.88  CONTEXT_LEN=32768  MAX_RUNNING=16  CHUNKED_PREFILL=4096
#   ATTN_BACKEND=flashinfer         triton if flashinfer fails at startup for this model
#   REASONING_PARSER=qwen3          thinking goes to reasoning_content; "" leaves it inline in content
#   EXTRA_ARGS="..."                any extra sglang flags, e.g. MTP speculative decoding:
#                                   "--speculative-algo NEXTN --speculative-num-steps 3
#                                    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
#   SERVER_ENV="VAR=..."            extra environment for the server process
#   MODEL_ID / MODEL_DIR / WORK_DIR  must match what setup_host_qwen3_27B.sh used
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3.5-27B-AWQ-4bit}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/${MODEL_ID##*/}}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-27b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.88}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SERVER_ENV="${SERVER_ENV:-}"
SESSION="${SESSION:-sglang}"  # same session as serve.sh: one server per GPU
CUDA_HOME_DIR="/usr/local/cuda-13.0"
LOG_DIR="$WORK_DIR/logs"
STARTUP_MIN=20  # 27B load + CUDA graph capture is slower than the 4B
URL="http://127.0.0.1:$PORT"
AUTH=()
if [[ -n "$API_KEY" ]]; then AUTH=(-H "Authorization: Bearer $API_KEY"); fi

running() { tmux has-session -t "$SESSION" 2>/dev/null; }
ready()   { curl -sf "${AUTH[@]}" "$URL/v1/models" >/dev/null 2>&1; }

summary() {
  local log="$LOG_DIR/latest.log"
  echo "--- startup facts (record these in the run log) ---"
  grep -m1 -iE "attention.?backend" "$log" || true
  grep -m1 -iE "quantization|compressed.?tensors|marlin" "$log" || true
  grep -m1 -E "max_total_num_tokens" "$log" || true
  grep -m1 -iE "mamba" "$log" || true
  grep -m1 -iE "cuda graph" "$log" || true
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo "--- test request (thinking off) ---"
  curl -s "${AUTH[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print('reply:', r['choices'][0]['message']['content'])"
  echo "Endpoint: http://$HOST:$PORT/v1   model name: $SERVED_NAME"
  echo "Harbor:   -m hosted_vllm/$SERVED_NAME --ak api_base='\"http://127.0.0.1:$PORT/v1\"'"
  tunnel
}

public_ip() {
  local ip
  ip=$(curl -s --max-time 3 https://ifconfig.me 2>/dev/null || true)
  [[ "$ip" =~ ^[0-9.]+$ ]] || ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo "${ip:-<gpu-host-ip>}"
}

tunnel() {
  echo "--- from your laptop ---"
  echo "ssh -N -L $PORT:127.0.0.1:$PORT $(whoami)@$(public_ip)"
  echo "  API: http://127.0.0.1:$PORT/v1   (python smoke_test.py --model $SERVED_NAME)"
  echo "  Add -i <key> or use your ssh host alias if that's how you normally log in."
}

start() {
  if running; then echo "A server is already running (tmux session '$SESSION'). Use: $0 restart"; return; fi
  [[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found at $MODEL_DIR; run setup_host_qwen3_27B.sh first."; exit 1; }
  [[ -d "$WORK_DIR/.venv" ]] || { echo "No venv at $WORK_DIR/.venv; run setup_host_qwen3_27B.sh first."; exit 1; }
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if (( used > 2048 )); then
    echo "GPU already has ${used} MiB in use; the 27B needs the whole card. Stop the other process first:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  local log; log="$LOG_DIR/server_27b_$(date +%Y%m%d_%H%M%S).log"
  local opt=""
  [[ -z "$API_KEY" ]] || opt+=" --api-key $API_KEY"
  [[ -z "$REASONING_PARSER" ]] || opt+=" --reasoning-parser $REASONING_PARSER"

  # The exact command is written to a file so every run is reproducible from the log dir.
  # No --quantization flag: SGLang reads the compressed-tensors config from the checkpoint.
  cat > "$WORK_DIR/.run_server_27b.sh" <<EOF
#!/usr/bin/env bash
source "$WORK_DIR/.venv/bin/activate"
export CUDA_HOME=$CUDA_HOME_DIR PATH=$CUDA_HOME_DIR/bin:\$PATH LD_LIBRARY_PATH=$CUDA_HOME_DIR/lib64:\${LD_LIBRARY_PATH:-}
${SERVER_ENV:+export $SERVER_ENV}
sglang serve "$MODEL_DIR" \\
  --served-model-name $SERVED_NAME \\
  --host $HOST --port $PORT \\
  --attention-backend $ATTN_BACKEND \\
  --sampling-backend flashinfer \\
  --tool-call-parser $TOOL_PARSER \\
  --context-length $CONTEXT_LEN \\
  --mem-fraction-static $MEM_FRACTION \\
  --max-running-requests $MAX_RUNNING \\
  --chunked-prefill-size $CHUNKED_PREFILL \\
  --enable-metrics \\
  --enable-cache-report \\
  --log-level info$opt $EXTRA_ARGS 2>&1 | tee "$log"
EOF
  chmod 700 "$WORK_DIR/.run_server_27b.sh"
  grep -v -- "--api-key" "$WORK_DIR/.run_server_27b.sh" > "${log%.log}.cmd" || true
  ln -sf "$log" "$LOG_DIR/latest.log"
  tmux new-session -d -s "$SESSION" "$WORK_DIR/.run_server_27b.sh"

  echo "Starting; log: $log"
  for i in $(seq 1 $((STARTUP_MIN * 12))); do
    if ready; then echo "Ready after ~$((i * 5)) s"; summary; return; fi
    if ! running; then
      echo "Server exited during startup. Last log lines:"; tail -n 40 "$log"
      echo "Out of memory? Try MEM_FRACTION=0.85, MAX_RUNNING=8 or CONTEXT_LEN=16384. Backend error? ATTN_BACKEND=triton."
      exit 1
    fi
    sleep 5
  done
  echo "Not ready after $STARTUP_MIN minutes; check: $0 logs"
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
  if ready; then
    echo "API: ready at $URL/v1, serving: $(curl -s "${AUTH[@]}" "$URL/v1/models" | python3 -c "import json,sys; print(', '.join(m['id'] for m in json.load(sys.stdin)['data']))")"
  else
    echo "API: not responding"
  fi
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
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
