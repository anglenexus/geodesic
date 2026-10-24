#!/usr/bin/env bash
# Start an SFT or GRPO run in the pinned Miles container on this box (one H200).
#
#   SAVE_DIR=/root/ckpt/sft_smoke1  bash scripts/run.sh sft
#   SAVE_DIR=/root/ckpt/grpo_pilot1 bash scripts/run.sh grpo    # needs scripts/agent_server.sh running
#
# Host side: docker run with the data disk mounted, then this same script runs again inside the container:
# start Ray and submit Miles' train_async.py (sft) or train.py (grpo) with configs/common.sh + configs/<mode>.sh.
# Container paths: /root/models, /root/ckpt, /root/miles (pinned source), /root/post_training (this folder).
set -euo pipefail
MODE=${1:?usage: run.sh sft|grpo}
HERE=$(cd "$(dirname "$0")/.." && pwd)                  # aees/post_training
source "$HERE/configs/pins.sh"
DATA=${DATA:-/data}

if [ "${2:-}" != inside ]; then
  exec docker run --rm --gpus all --ipc=host --network host --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$DATA/models:/root/models" -v "$DATA/ckpt:/root/ckpt" -v "$DATA/src/miles:/root/miles" \
    -v "$HERE:/root/post_training" \
    -e SAVE_DIR -e SFT_DATA -e PROMPT_DATA -e NUM_ROLLOUT -e INIT_ADAPTER -e SGLANG_MEM -e AGENT_SERVER_URL \
    -e MODEL -e MODEL_TYPE -e MAX_LEN \
    "$MILES_IMAGE" bash /root/post_training/scripts/run.sh "$MODE" inside
fi

# --- inside the container ---
pip install -e /root/miles --no-deps -q                 # the pinned source mounted over the image's copy
source /root/post_training/configs/common.sh
source "/root/post_training/configs/$MODE.sh"
case "$MODE" in
  sft)  SCRIPT=train_async.py; MODE_ARGS=("${SFT_ARGS[@]}" "${OPTIM_ARGS[@]}" "${SAVE_ARGS[@]}") ;;   # as Miles' SFT script
  grpo) SCRIPT=train.py;       MODE_ARGS=("${GRPO_ARGS[@]}" "${OPTIM_ARGS[@]}" "${SAVE_ARGS[@]}") ;; # as Miles' Harbor examples
  *) echo "unknown mode: $MODE (use sft or grpo)"; exit 2 ;;
esac
mkdir -p "$SAVE_DIR"
cp /root/post_training/configs/{pins,common,$MODE}.sh "$SAVE_DIR/"     # the run's exact config, kept with it

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats
HARBOR_EXAMPLE=/root/miles/examples/swe-agent-harbor-docker     # swe_agent_function.py, generate.py (GRPO)
# AGENT_TRIAL_TIMEOUT 12 h: counts queue time behind the 16 running trials; Harbor's own timeouts fire first
ENV='{"env_vars": {"PYTHONPATH": "/root/miles:/root/Megatron-LM:/root/post_training:'"$HARBOR_EXAMPLE"'",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTHONUNBUFFERED": "1", "NCCL_NVLS_ENABLE": "0",
      "AGENT_SERVER_URL": "'"${AGENT_SERVER_URL:-http://127.0.0.1:11000}"'",
      "AGENT_TRIAL_TIMEOUT": "43200"}}'
ray job submit --address http://127.0.0.1:8265 --runtime-env-json "$ENV" -- \
  python3 "/root/miles/$SCRIPT" "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${LORA_ARGS[@]}" "${PERF_ARGS[@]}" \
  "${CLUSTER_ARGS[@]}" "${MODE_ARGS[@]}" 2>&1 | tee -a "$SAVE_DIR/train.log"
