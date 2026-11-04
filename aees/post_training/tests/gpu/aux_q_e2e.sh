#!/usr/bin/env bash
# GPU end-to-end test of --opd-baseline exact with a LoRA teacher (--opd-teacher-lora-path; grpo/aux_reward.patch) on any 1-GPU box with setup.sh steps 2-4, 6, 7
# done for Qwen/Qwen3.5-4B (~25 min: teacher + 2 conversions + one train-only step). Miles' real path: teacher pass,
# policy pass, advantages; then tests/gpu/aux_q_check.py against an independent HF computation. Prints AUX_Q_E2E_OK.
#
#   bash tests/gpu/aux_q_e2e.sh                                   # 4B, a random PEFT LoRA teacher (any GPU)
#   MODEL=Qwen3.6-27B MODEL_TYPE=qwen3.6-27B LORA_RANK=64 TEACHER_LORA=/root/models/qfunc_adapter \
#     REF_TEACHER=/root/models/Qwen3.6-27B-qfunc bash tests/gpu/aux_q_e2e.sh   # T-AUX1 on the H200: the real Q adapter
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)                # aees/post_training
source "$HERE/configs/pins.sh"
DATA=${DATA:-/data}
COEF=${COEF:-0.5}
mkdir -p "$DATA/ckpt"
git -C "$DATA/src/miles" apply --check -R "$HERE/grpo/aux_reward.patch" 2>/dev/null \
  || { git -C "$DATA/src/miles" checkout -- . && git -C "$DATA/src/miles" clean -fdq miles && git -C "$DATA/src/miles" apply "$HERE/grpo/aux_reward.patch"; }
sudo docker run --rm --gpus all --ipc=host --network host --shm-size 32g --ulimit memlock=-1 \
  -v "$DATA/models:/root/models" -v "$DATA/ckpt:/root/ckpt" -v "$DATA/src/miles:/root/miles" -v "$HERE:/root/post_training" \
  -e COEF="$COEF" -e MODEL="${MODEL:-Qwen3.5-4B}" -e MODEL_TYPE="${MODEL_TYPE:-qwen3.5-4B}" -e LORA_RANK="${LORA_RANK:-8}" \
  -e TEACHER_LORA="${TEACHER_LORA:-}" -e REF_TEACHER="${REF_TEACHER:-}" "$MILES_IMAGE" bash -c '
set -euo pipefail
pip install -e /root/miles --no-deps -q
cd /root/post_training/tests/gpu
M=/root/models/$MODEL; W=/root/ckpt/auxtest_$MODEL
L=${TEACHER_LORA:-/root/models/$MODEL-lora-teacher}; R=${REF_TEACHER:-$L/merged}
[ -f $L/adapter_model.safetensors ] || python3 aux_q_setup.py lora $M $L
read -ra MODEL_ARGS <<< "$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py $MODEL_TYPE)"
for m in $M; do
  [ -d ${m}_torch_dist ] || (cd /root/miles && PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 1 \
    tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint $m --save ${m}_torch_dist)
done
mkdir -p $W
PYTHONPATH=/root/miles python3 aux_q_setup.py rollout $M $W/rollout_0.pt
printf "{\"prompt\": \"x\"}\n{\"prompt\": \"y\"}\n" > $W/prompts.jsonl
rm -f $W/train_0.pt
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats > /dev/null
ray job submit --address http://127.0.0.1:8265 --runtime-env-json "{\"env_vars\": {\"PYTHONPATH\": \"/root/miles:/root/Megatron-LM\", \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\", \"MILES_AUX_Q_DIR\": \"$W/aux_q\"}}" -- \
  python3 /root/miles/train.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint $M --load ${M}_torch_dist --megatron-to-hf-mode raw --no-load-rng --finetune --no-load-optim \
  --lora-rank $LORA_RANK --lora-alpha $LORA_RANK --lora-dropout 0.0 --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --no-gradient-accumulation-fusion --lora-base-cpu-backup \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 1024 --log-probs-backend fused \
  --attention-backend flash --attention-dropout 0.0 --hidden-dropout 0.0 --calculate-per-token-loss \
  --colocate --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 \
  --prompt-data $W/prompts.jsonl --input-key prompt --num-rollout 1 \
  --rollout-batch-size 2 --n-samples-per-prompt 3 --global-batch-size 6 --rollout-max-response-len 1024 \
  --debug-train-only --load-debug-rollout-data "$W/rollout_{rollout_id}.pt" --save-debug-train-data "$W/train_{rollout_id}.pt" \
  --advantage-estimator grpo --eps-clip 0.2 --loss-type policy_loss --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 1e-5 --lr-decay-style constant --clip-grad 0 \
  --use-opd --opd-type megatron --opd-teacher-lora-path $L --opd-kl-coef $COEF --opd-baseline exact \
  2>&1 | tail -40
nvidia-smi --query-gpu=memory.used --format=csv,noheader; free -g | head -2
python3 aux_q_check.py $W/train_0.pt $W/rollout_0.pt $M $R $COEF
'
