#!/usr/bin/env bash
# On the 8xH100 host (align_tmax_plan.md, Experiment 1): 4 SGLang engines, each on 2 GPUs (TP2), for rollouts with
# thinking on. Engine i serves 127.0.0.1:3000<i>, which every box reaches through the tunnels (links.sh: -R 30000-30015).
# Context 65,536 in total, as Tmax's eval: a call is refused once prompt + 16,384 output tokens would exceed it.
# No LoRA and no Miles: the base model, served the way the agents will call it (OpenAI API, model name "model").
#
#   bash scripts/phase1_serve.sh start | status | stop
# Logs: $DATA/monitor/$RUN/sglang_<i>.log. Env: ENGINES (4), TP (2), CONTEXT (65536), MAX_RUNNING (16 per engine;
# 40 trials / 4 engines = 10, with room), MEM (0.88).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
source "$HERE/configs/pins.sh"
DATA=${DATA:-/data}
RUN=${RUN:?set RUN}
ENGINES=${ENGINES:-4}
TP=${TP:-2}
CONTEXT=${CONTEXT:-65536}                                # Tmax's vLLM --max-model-len: prompt + output <= 65,536
MAX_RUNNING=${MAX_RUNNING:-16}
MEM=${MEM:-0.88}
MODEL=${MODEL:-Qwen3.6-27B}

engine_up() { curl -sf --max-time 5 "127.0.0.1:$((30000 + $1))/health_generate" >/dev/null 2>&1; }

case "${1:-}" in
start)
  mkdir -p "$DATA/monitor/$RUN" "$DATA/cache"
  ngpu=$(nvidia-smi -L | grep -c GPU)
  [ "$ngpu" -ge $((ENGINES * TP)) ] || { echo "need $((ENGINES * TP)) GPUs, found $ngpu"; exit 1; }
  for i in $(seq 1 "$ENGINES"); do
    gpus=$(seq $(((i - 1) * TP)) $((i * TP - 1)) | paste -sd, -)
    docker rm -f "sglang$i" >/dev/null 2>&1 || true
    docker run -d --name "sglang$i" --gpus "\"device=$gpus\"" --ipc=host --network host --shm-size 16g \
      --ulimit memlock=-1 -v "$DATA/models:/root/models" -v "$DATA/cache:/root/.cache" -v "$DATA/monitor/$RUN:/root/logs" \
      "$MILES_IMAGE" bash -c "python3 -m sglang.launch_server --model-path /root/models/$MODEL --served-model-name model \
        --host 127.0.0.1 --port $((30000 + i)) --tp-size $TP --context-length $CONTEXT --dtype bfloat16 \
        --mem-fraction-static $MEM --max-running-requests $MAX_RUNNING --reasoning-parser qwen3 \
        --enable-metrics --enable-cache-report > /root/logs/sglang_$i.log 2>&1" >/dev/null
    echo "engine $i: GPUs $gpus, port $((30000 + i))"
  done
  for _ in $(seq 180); do                                # first start compiles kernels: allow ~15 min
    up=0; for i in $(seq 1 "$ENGINES"); do engine_up "$i" && up=$((up + 1)); done
    [ "$up" = "$ENGINES" ] && break
    for i in $(seq 1 "$ENGINES"); do docker ps -q -f "name=^sglang$i$" | grep -q . || { echo "engine $i exited:"; tail -20 "$DATA/monitor/$RUN/sglang_$i.log"; exit 1; }; done
    sleep 5
  done
  bash "$0" status ;;
status)
  for i in $(seq 1 "$ENGINES"); do
    state=$(engine_up "$i" && echo up || echo down)
    cap=$(grep -oE "max_running_requests is capped to [0-9]+|max_total_num_tokens=[0-9]+" "$DATA/monitor/$RUN/sglang_$i.log" 2>/dev/null | tr '\n' ' ' || true)
    echo "engine $i (port $((30000 + i))): $state ${cap}"
  done ;;
stop)
  for i in $(seq 1 "$ENGINES"); do docker rm -f "sglang$i" >/dev/null 2>&1 || true; done; echo "engines stopped" ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
