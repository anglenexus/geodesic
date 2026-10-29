#!/usr/bin/env bash
# Overlap prototype on the H200 (PLAN.md §5b): can SGLang serve agent rollouts while LoRA training runs on
# the same GPU, and at what cost to each? ~60 GPU-minutes. Everything runs in one pinned Miles container;
# the MPS daemon runs on the host and the container shares its pipe folder. Results: $DATA/proto/$RUN.
#
#   STEPS="1 2 3 4 6 7" MODELS="Qwen/Qwen3.6-27B Qwen/Qwen3.6-27B-FP8" bash setup.sh    # once
#   bash proto/bench.sh                                   # all phases (proto/kickoff.sh does both)
#   PHASES="train_alone serve_fp8" bash proto/bench.sh    # a subset, e.g. to re-run one
#
# Phases (minutes):  pcie (+ the random LoRA adapter, 1) | train_alone, train_alone_offload (3 + 3) | serve_bf16 (5) |
#   serve_fp8_32, serve_fp8 (4 + 4) | co_timeslice (CO_MIN + 2) | serve_fp8_mps (3) | co_mps, co_mps70 (2 x (CO_MIN + 2))
# Each phase folder: phase.json, agents.jsonl, train.jsonl, gpu.csv, dcgm.txt, sglang.log (see proto/report.py).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)                   # aees/post_training
source "$HERE/configs/pins.sh"
DATA=${DATA:-/data}
RUN=${RUN:-$(date +%Y%m%d_%H%M)}
OUT=$DATA/proto/$RUN                                      # host path; /root/proto_out/$RUN in the container
C=proto                                                   # container name
BF16=/root/models/Qwen3.6-27B
FP8=/root/models/Qwen3.6-27B-FP8
SERVE_MEM_ALONE=${SERVE_MEM_ALONE:-0.85}                  # as in production (configs/grpo.sh)
SERVE_MEM_CO=${SERVE_MEM_CO:-0.36}                        # FP8 server next to the trainer: ~51 GB
AGENTS=${AGENTS:-16}
TRAIN_MIN=${TRAIN_MIN:-3}
SERVE_MIN=${SERVE_MIN:-4}
CO_MIN=${CO_MIN:-5}                                       # training inside each co-run phase
CO_WARMUP_S=${CO_WARMUP_S:-60}                            # agents alone before the trainer starts
MPS_CAP=${MPS_CAP:-70}                                    # co_mps70: trainer's share of SMs under MPS
PHASES=${PHASES:-"pcie train_alone train_alone_offload serve_bf16 serve_fp8_32 serve_fp8 co_timeslice serve_fp8_mps co_mps co_mps70"}
DCGM_IMAGE=${DCGM_IMAGE:-nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04}
MPS_PIPE=${MPS_PIPE:-/tmp/proto_mps}                     # host folder, mounted at /tmp/mps in the container
LORA=1                                                    # cleared if SGLang can't start with the adapter
mkdir -p "$OUT" "$MPS_PIPE" "$MPS_PIPE.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bench.log"; }
now() { date +%s.%N | grep -v N || date +%s; }               # epoch seconds (BSD date has no %N)

# --- container, samplers, MPS ------------------------------------------------------------------------
start_container() {
  docker rm -f "$C" >/dev/null 2>&1 || true
  docker run -d --name "$C" --gpus all --ipc=host --network host --shm-size 32g --ulimit memlock=-1 \
    -v "$DATA/models:/root/models" -v "$DATA/proto:/root/proto_out" -v "$HERE:/root/post_training" \
    -v "$MPS_PIPE:/tmp/mps" -v "$MPS_PIPE.log:/tmp/mps_log" -w /root/post_training \
    -e PYTHONPATH=/root/post_training \
    "$MILES_IMAGE" sleep infinity >/dev/null
  cp "$DATA/models/Qwen3.6-27B/config.json" "$OUT/config.json"
  if docker run -d --name proto_dcgm --gpus all --cap-add SYS_ADMIN --network host "$DCGM_IMAGE" >/dev/null 2>&1; then
    sleep 5; log "DCGM profiling counters: on"
  else
    log "DCGM unavailable ($DCGM_IMAGE); GPU counters from nvidia-smi only"
  fi
}
samplers_start() {   # per phase: nvidia-smi every second; DCGM SM / tensor / DRAM activity + PCIe bytes
  local dir=$1
  ( while :; do
      echo "$(now),$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits | head -1)"
      sleep 1
    done ) > "$dir/gpu.csv" 2>/dev/null &
  SAMPLERS=$!
  if docker ps -q -f name=proto_dcgm | grep -q .; then
    ( docker exec proto_dcgm dcgmi dmon -e 1002,1004,1005,1009,1010 -d 1000 2>&1 \
        | while IFS= read -r l; do echo "$(now) $l"; done ) > "$dir/dcgm.txt" &
    SAMPLERS="$SAMPLERS $!"
  fi
}
samplers_stop() {
  kill $SAMPLERS 2>/dev/null || true
  wait $SAMPLERS 2>/dev/null || true                      # reap quietly (no "Terminated" lines)
  pkill -f "dcgmi dmon" 2>/dev/null || true
}

# MPS: one control daemon on the host. Processes started with CUDA_MPS_PIPE_DIRECTORY pointing at its pipe
# folder become MPS clients and share one GPU context, so their kernels run at the same time. Without it,
# each process has its own context and the GPU time-slices between them.
mps_on()  { sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_PIPE.log" nvidia-cuda-mps-control -d; }
mps_off() { echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control 2>/dev/null || true; }
mps_env() { [ "$1" = 1 ] && echo "-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log"; true; }

# --- phases ------------------------------------------------------------------------------------------
phase_begin() { PDIR=$OUT/$1; mkdir -p "$PDIR"; P_NAME=$1; P_T0=$(now); samplers_start "$PDIR"; log "phase $1"; }
phase_end() {         # phase_end <serve: bf16|fp8|null> <mps 0|1> <agents> <train json|null> <co 0|1> [mps pct]
  samplers_stop
  printf '{"name":"%s","serve":%s,"mps":%s,"agents":%s,"train":%s,"co":%s,"mps_pct":%s,"lora":%s,"t0":%s,"t1":%s}\n' \
    "$P_NAME" "$1" "$([ "$2" = 1 ] && echo true || echo false)" "$3" "$4" "$([ "$5" = 1 ] && echo true || echo false)" \
    "${6:-null}" "$([ "$LORA" = 1 ] && echo true || echo false)" "$P_T0" "$(now)" > "$PDIR/phase.json"
}

serve_stop() {
  docker exec "$C" pkill -f sglang.launch_server 2>/dev/null || true
  for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return
    sleep 2
  done
  docker exec "$C" pkill -9 -f sglang 2>/dev/null || true; sleep 5
}
serve() {             # serve <bf16|fp8> <mem fraction> <mps 0|1> <log>: start SGLang, wait until it answers
  local model=$BF16 extra=(--max-running-requests 48) try   # tries: with the adapter, without, without FP8 KV
  [ "$1" = fp8 ] && { model=$FP8; extra+=(--kv-cache-dtype fp8_e4m3); }
  serve_stop
  for try in lora plain plain-kv16; do
    [ "$try" = lora ] && [ "$LORA" = 0 ] && continue
    if [ "$try" = plain-kv16 ]; then [ "$1" = fp8 ] || continue; extra=(--max-running-requests 48); fi
    local lora=()
    [ "$try" = lora ] && lora=(--enable-lora --max-lora-rank 64 --lora-backend triton
      --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
      --lora-paths "proto=/root/proto_out/$RUN/adapter")
    # shellcheck disable=SC2046
    docker exec -d $(mps_env "$3") "$C" bash -c "python3 -m sglang.launch_server --model-path $model \
      --served-model-name qwen3.6-27b --host 127.0.0.1 --port 30000 --mem-fraction-static $2 \
      --context-length 57344 --enable-cache-report --decode-log-interval 40 ${extra[*]} ${lora[*]:-} \
      > ${4/#$OUT//root/proto_out/$RUN} 2>&1"               # the log's path inside the container
    for _ in $(seq 180); do
      curl -sf http://127.0.0.1:30000/health_generate >/dev/null 2>&1 && { log "SGLang up: $1, mem $2, mps $3, $try"; return 0; }
      docker exec "$C" pgrep -f sglang.launch_server >/dev/null || break
      sleep 5
    done
    log "SGLang failed to start ($1, $try); last lines:"; tail -5 "$4" | tee -a "$OUT/bench.log"
    serve_stop
    LORA=0
  done
  return 1
}
agents() {            # agents <n> <minutes>: run the agent load in the foreground (HTTP only, no CUDA)
  local lora=(); [ "$LORA" = 1 ] && lora=(--lora proto)
  docker exec "$C" python3 -m proto.agent_load --profile proto/load_profile.json --agents "$1" \
    --minutes "$2" --out "/root/proto_out/$RUN/$P_NAME/agents.jsonl" ${lora[@]+"${lora[@]}"}
}
train() {             # train <minutes> <mps 0|1> <thread pct or ""> [proxy args...]: retries at 32k on OOM
  local min=$1 mps=$2 pct=$3; shift 3
  local env=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)          # less fragmentation next to SGLang
  [ -n "$pct" ] && env+=(-e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct")
  # shellcheck disable=SC2046
  docker exec $(mps_env "$mps") ${env[@]+"${env[@]}"} "$C" python3 -m proto.train_proxy --config "$BF16/config.json" \
    --minutes "$min" --out "/root/proto_out/$RUN/$P_NAME/train.jsonl" "$@" && return 0
  [ $? = 3 ] || return 1
  log "trainer out of memory at 48k; retrying at 32k"
  # shellcheck disable=SC2046
  docker exec $(mps_env "$mps") ${env[@]+"${env[@]}"} "$C" python3 -m proto.train_proxy --config "$BF16/config.json" \
    --minutes "$min" --out "/root/proto_out/$RUN/$P_NAME/train.jsonl" --seq 32768 "$@" || true
}

# Co-run: agents keep the server busy for CO_MIN + 2 minutes and the trainer runs in the middle CO_MIN, so
# each phase has calls before/after training ("alone") and during it ("with training"), on the same server.
#   Time-slicing (mps 0): two CUDA contexts; the GPU runs one context's kernels at a time and switches
#     between them, so a decode step waits behind training kernels.
#   MPS (mps 1): both processes are clients of one MPS server and share a single context, so decode and
#     training kernels run at the same time on different SMs. HBM bandwidth and L2 stay shared. With a
#     thread percentage, the trainer may use only that share of the SMs, which leaves the rest for decode.
co_phase() {          # co_phase <name> <mps 0|1> <trainer's SM % under MPS, or "">
  local a hot=""
  phase_begin "$1"
  agents "$AGENTS" $((CO_MIN + 2)) "$2" & a=$!
  sleep "$CO_WARMUP_S"
  if [ "$1" = co_mps70 ] && [ "$LORA" = 1 ]; then          # adapter pushes while both run
    docker exec "$C" python3 -m proto.agent_load --hot-load "/root/proto_out/$RUN/adapter" --times 5 --gap 45 \
      --out "/root/proto_out/$RUN/$1/hotload.jsonl" & hot=$!
  fi
  train "$CO_MIN" "$2" "$3" --offload || log "trainer failed in $1"   # FP8 server + offloaded trainer: fits
  wait "$a" || true
  [ -z "$hot" ] || wait "$hot" || true
  phase_end '"fp8"' "$2" "$AGENTS" '{"offload":true}' 1 "${3:-null}"
}
alone_phase() {       # alone_phase <name> <serve> <mps> <agents>: the server alone under agent load
  phase_begin "$1"
  agents "$4" $((SERVE_MIN + 1)) "$3" || log "agents failed in $1"
  phase_end "\"$2\"" "$3" "$4" null 0
}
has() { [[ " $PHASES " == *" $1 "* ]]; }

# --- run -------------------------------------------------------------------------------------------
cleanup() { samplers_stop; serve_stop; mps_off; docker rm -f "$C" proto_dcgm >/dev/null 2>&1 || true; }
trap cleanup EXIT
log "run $RUN: phases $PHASES"
start_container
if has pcie; then
  phase_begin pcie
  docker exec "$C" python3 -m proto.train_proxy --config "$BF16/config.json" --pcie-only \
    --out "/root/proto_out/$RUN/pcie/train.jsonl" || log "pcie test failed"
  phase_end null 0 0 null 0
fi
[ -f "$OUT/adapter/adapter_config.json" ] || docker exec "$C" python3 -m proto.adapter "$BF16" "/root/proto_out/$RUN/adapter" \
  || { log "adapter failed: serving without LoRA"; LORA=0; }
for t in train_alone train_alone_offload; do
  has $t || continue
  phase_begin $t
  if [ $t = train_alone ]; then train "$TRAIN_MIN" 0 "" || true; else train "$TRAIN_MIN" 0 "" --offload || true; fi
  phase_end null 0 0 "{\"offload\":$([ $t = train_alone ] && echo false || echo true)}" 0
done
if has serve_bf16; then
  mkdir -p "$OUT/serve_bf16"
  serve bf16 "$SERVE_MEM_ALONE" 0 "$OUT/serve_bf16/sglang.log" && alone_phase serve_bf16 bf16 0 "$AGENTS"
fi
if has serve_fp8_32 || has serve_fp8 || has co_timeslice; then
  mkdir -p "$OUT/serve_fp8"
  if serve fp8 "$SERVE_MEM_CO" 0 "$OUT/serve_fp8/sglang.log"; then
    if has serve_fp8_32; then alone_phase serve_fp8_32 fp8 0 32; fi
    if has serve_fp8; then alone_phase serve_fp8 fp8 0 "$AGENTS"; fi
    if has co_timeslice; then co_phase co_timeslice 0 ""; fi
  fi
fi
if has serve_fp8_mps || has co_mps || has co_mps70; then
  serve_stop
  mps_on
  mkdir -p "$OUT/serve_fp8_mps"
  if serve fp8 "$SERVE_MEM_CO" 1 "$OUT/serve_fp8_mps/sglang.log"; then
    if has serve_fp8_mps; then alone_phase serve_fp8_mps fp8 1 "$AGENTS"; fi
    if has co_mps; then co_phase co_mps 1 ""; fi
    if has co_mps70; then co_phase co_mps70 1 "$MPS_CAP"; fi
  fi
fi
log "done: $OUT  (laptop: bash proto/fetch.sh)"
