# Flags shared by SFT (configs/sft.sh) and GRPO (configs/grpo.sh), as bash arrays.
# Sourced inside the Miles container; scripts/run.sh mounts /root/models and /root/post_training.
# Native LoRA on Qwen3.6-27B, one H200, trainer and SGLang take turns on the GPU (PLAN.md §1, §3d).

MODEL=${MODEL:-Qwen3.6-27B}                      # Qwen3.5-4B for the Phase 1 smoke test
MODEL_TYPE=${MODEL_TYPE:-qwen3.6-27B}            # Miles' name for it: scripts/models/<type>.py
MAX_LEN=${MAX_LEN:-49152}                        # 48k training context (D10)

read -ra MODEL_ARGS <<< "$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py "$MODEL_TYPE")"

CKPT_ARGS=(
  --hf-checkpoint "/root/models/$MODEL"          # tokenizer, config, weights for SGLang
  --load "/root/models/${MODEL}_torch_dist"      # base weights for Megatron (native LoRA needs torch_dist)
  --megatron-to-hf-mode raw                      # the native LoRA path
  --no-load-rng --finetune                       # base weights only (no base step count or RNG)
)
[ -n "${RESUME_FROM:-}" ] || CKPT_ARGS+=(--no-load-optim)   # fresh Adam, unless resuming (configs/grpo.sh)
LORA_ARGS=(
  --lora-rank 64 --lora-alpha 64 --lora-dropout 0.0                           # D5
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj    # D6: 318.7M parameters
  --no-gradient-accumulation-fusion              # required by native LoRA
  --lora-base-cpu-backup                         # base weights stay in host RAM between phases (§1)
)
PERF_ARGS=(
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1   # one GPU
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_LEN"   # packing, at most 48k tokens per pass (§2)
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1   # keep only layer inputs (§1)
  --log-probs-backend fused                      # log-probs without extra fp32 copies of the 24 GB logits
  --attention-backend flash --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32
  --attention-dropout 0.0 --hidden-dropout 0.0
  --calculate-per-token-loss                     # token mean (§3b)
)
CLUSTER_ARGS=(
  --colocate                                     # required by native LoRA: trainer and SGLang share the GPU
  --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1
)
