#!/usr/bin/env bash
# Back up one run's outputs (LoRA adapters, launch config, optional trial archives) to a private
# Hugging Face model repo or an S3 prefix. Run it by hand for the runs worth keeping (not every smoke
# test). Re-running uploads only what's missing or changed. Never uploads base or merged model weights.
#
#   bash scripts/backup.sh /data/ckpt/grpo_pilot1 hf://<org>/<repo>       # one folder per run in the repo
#   bash scripts/backup.sh /data/ckpt/grpo_pilot1 s3://<bucket>/<prefix>  # <prefix>/<run>/...
#
#   WITH_OPTIMIZER=1   also upload optimizer/scheduler state (training_state_rank*.pt, ~4 GB per checkpoint),
#                      needed only to resume training on another machine
#   TRIALS=<dir>       also archive the run's Harbor trial folders, one .tar.gz per job folder
#
# Auth comes from `hf auth login` (write token) or the usual AWS credentials; nothing is stored here.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,13p "$0"; exit 2; }
RUN_DIR=$(realpath "$1"); DEST=${2%/}
RUN=$(basename "$RUN_DIR")
[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR"; exit 1; }

# What goes up: adapter folders of every saved iteration (Miles: iter_*/adapter/), HF-format adapters
# from --save-hf exports (*/adapter/ next to merged weights, whose *.safetensors shards stay behind),
# top-level launch config/logs, and trial archives.
include=("iter_*/adapter/*" "*/adapter/adapter_*" "*.sh" "*.json" "*.yaml" "*.log" "trials/*.tar.gz")
exclude=()
[ "${WITH_OPTIMIZER:-0}" = 1 ] || exclude+=("*/training_state_rank*.pt")

if [ -n "${TRIALS:-}" ]; then
  mkdir -p "$RUN_DIR/trials"
  for job in "$TRIALS"/*/; do
    out="$RUN_DIR/trials/$(basename "$job").tar.gz"
    [ -f "$out" ] || tar -C "$(dirname "$job")" -czf "$out" "$(basename "$job")"
  done
fi

case "$DEST" in
  hf://*)
    repo=${DEST#hf://}
    uvx --from huggingface_hub hf repo create "$repo" --repo-type model --private --exist-ok
    args=()
    for p in "${include[@]}"; do args+=(--include "$p"); done
    for p in ${exclude[@]+"${exclude[@]}"}; do args+=(--exclude "$p"); done
    uvx --from huggingface_hub hf upload "$repo" "$RUN_DIR" "$RUN" --repo-type model \
      --commit-message "backup $RUN $(date -u +%FT%TZ)" "${args[@]}"
    ;;
  s3://*)
    args=(--exclude "*")
    for p in "${include[@]}"; do args+=(--include "$p"); done
    for p in ${exclude[@]+"${exclude[@]}"}; do args+=(--exclude "$p"); done
    aws s3 sync "$RUN_DIR" "$DEST/$RUN" "${args[@]}"
    ;;
  *) echo "destination must start with hf:// or s3://"; exit 2 ;;
esac
echo "backed up $RUN_DIR -> $DEST ($RUN)"
