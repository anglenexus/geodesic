#!/usr/bin/env bash
# setup_host_qwen3_27B.sh - one-time setup for serving Qwen3.5-27B (4-bit AWQ) on one A100-40GB.
#
# Runs setup_host.sh (driver, CUDA 13, uv venv, SGLang + FlashInfer) with the 27B checkpoint, then
# checks what is specific to this model: free disk, SGLang support for the Qwen3.5 architecture,
# the 4-bit quantization config, and that the weights leave room for the cache on this GPU.
#
# Usage:  bash setup_host_qwen3_27B.sh      then: ./serve_qwen3_27B.sh start
# Safe to re-run. If setup_host.sh installs a GPU driver, reboot and run this script again.
#
# Why this checkpoint: BF16 weights are ~54 GB and the official GPTQ-Int4 is ~30 GB (it keeps the
# attention layers in BF16), which leaves almost no KV cache on 40 GB. This W4A16 build is ~20 GB.
#
# Options (env vars):
#   MODEL_ID=cyankiwi/Qwen3.5-27B-AWQ-4bit   4-bit checkpoint (compressed-tensors, runs on Marlin kernels)
#   SGLANG_VERSION=0.5.21                     "" = latest; use that if the architecture check fails
#   WORK_DIR=~/qwen-serve  MODEL_DIR=~/models/<repo name>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export MODEL_ID="${MODEL_ID:-cyankiwi/Qwen3.5-27B-AWQ-4bit}"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/${MODEL_ID##*/}}"
export WORK_DIR="${WORK_DIR:-$HOME/qwen-serve}"
export SGLANG_VERSION="${SGLANG_VERSION-0.5.21}"
ARCH="Qwen3_5ForConditionalGeneration"
NEED_DISK_GB=25

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }

# 0. Disk: ~20 GB of weights plus the venv
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  mkdir -p "$(dirname "$MODEL_DIR")"
  FREE_GB=$(df -Pk "$(dirname "$MODEL_DIR")" | awk 'NR==2 {print int($4 / 1048576)}')
  (( FREE_GB >= NEED_DISK_GB )) || die "Only ${FREE_GB} GB free under $(dirname "$MODEL_DIR"); need ~${NEED_DISK_GB} GB."
fi

# 1-8. Shared stack + model download
bash "$HERE/setup_host.sh"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "setup_host.sh stopped before the model download (driver install?). Reboot if asked, then re-run $0."
  exit 0
fi

# shellcheck disable=SC1091
source "$WORK_DIR/.venv/bin/activate"

# 9. SGLang must know the Qwen3.5 architecture
log "SGLang support for $ARCH"
SGL_DIR=$(python -c "import os, sglang; print(os.path.dirname(sglang.__file__))")
if grep -rqs "$ARCH" "$SGL_DIR/srt/models"; then
  echo "$ARCH found in sglang $(python -c 'import sglang; print(sglang.__version__)')"
else
  die "This SGLang has no $ARCH. Re-run with SGLANG_VERSION=\"\" bash $0 to install the latest release."
fi
python -c "import compressed_tensors" 2>/dev/null || uv pip install compressed-tensors

# 10. Checkpoint and GPU fit
log "Checkpoint and GPU memory"
python - "$MODEL_DIR" <<'EOF'
import json, pathlib, subprocess, sys
d = pathlib.Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
q = cfg.get("quantization_config", {})
print("architecture:", cfg["architectures"][0], "| quant_method:", q.get("quant_method"), "| format:", q.get("format"))
if not q:
    sys.exit("No quantization_config: this looks like the BF16 checkpoint, which does not fit on 40 GB.")
weights_gb = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 2**30
gpu_gb = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).split()[0]) / 1024
print(f"weights {weights_gb:.1f} GiB on a {gpu_gb:.1f} GiB GPU")
# serve_qwen3_27B.sh uses MEM_FRACTION 0.88; want >= ~8 GiB left for KV + linear-attention state
left = gpu_gb * 0.88 - weights_gb
print(f"~{left:.1f} GiB left for the cache at mem fraction 0.88")
if left < 8:
    print("WARNING: tight. Expect few concurrent requests; lower CONTEXT_LEN or MAX_RUNNING in serve_qwen3_27B.sh.")
EOF

log "Setup complete. Start serving with: ./serve_qwen3_27B.sh start"
