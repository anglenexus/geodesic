# SFT set (PLAN.md §3d; optional warm start, D15). Sourced after configs/common.sh by scripts/run.sh sft.
# Training only: no SGLang (--debug-train-only). Chats are rendered as GRPO sees them by sft/rollout.py.

SFT_DATA=${SFT_DATA:-/root/post_training/data/sft_qwen35_27b.jsonl}   # from sft/export_trajectories.py
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR to a new folder per run, e.g. /root/ckpt/sft_smoke1}

SFT_ARGS=(
  --rollout-function-path sft.rollout.generate_rollout    # our rollout, not Miles' built-in one
  --prompt-data "$SFT_DATA" --input-key messages
  --rollout-shuffle --num-epoch 2                          # 2 epochs (§3d)
  --rollout-batch-size 16 --global-batch-size 16           # 16 chats per optimizer step (§3d)
  --loss-type sft_loss --disable-compute-advantages-and-returns
  --debug-train-only                                       # no SGLang engine
)
OPTIM_ARGS=(
  --optimizer adam --lr 1e-4 --lr-decay-style constant     # §3d
  --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --weight-decay 0
  --clip-grad 0                                            # off (§3d); the grad norm is still logged
)
SAVE_ARGS=(
  --save "$SAVE_DIR"
  --save-interval 1000000                                  # i.e. only at each epoch end and the last step
)
